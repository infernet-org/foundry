#!/bin/bash
# ==============================================================================
# Foundry: Host Kernel Tuning for Inference Latency
# ==============================================================================
# Run this ONCE on the Docker host to optimize kernel parameters for LLM
# inference workloads. Requires root/sudo.
#
# Usage:
#   sudo ./scripts/host-setup.sh
#
# These changes are NOT persistent across reboots. To make them permanent,
# add them to /etc/sysctl.d/99-foundry.conf and update GRUB for hugepages.
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[foundry-host]${NC} $*"; }
warn() { echo -e "${YELLOW}[foundry-host]${NC} $*" >&2; }
ok()   { echo -e "${GREEN}[foundry-host]${NC} $*"; }

if [ "$(id -u)" -ne 0 ]; then
    echo -e "${RED}ERROR: This script must be run as root (sudo)${NC}" >&2
    exit 1
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Foundry Host Kernel Tuning${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# ==============================================================================
# Memory: Reduce swappiness to keep model weights in RAM
# ==============================================================================
CURRENT_SWAPPINESS=$(cat /proc/sys/vm/swappiness)
log "vm.swappiness: ${CURRENT_SWAPPINESS} -> 0"
sysctl -w vm.swappiness=0 > /dev/null
ok "vm.swappiness = 0 (model weights strictly stay in RAM)"

# ==============================================================================
# Memory: Disable NUMA balancing (prevents random latency spikes)
# ==============================================================================
if [ -f /proc/sys/kernel/numa_balancing ]; then
    CURRENT_NUMA=$(cat /proc/sys/kernel/numa_balancing)
    log "kernel.numa_balancing: ${CURRENT_NUMA} -> 0"
    sysctl -w kernel.numa_balancing=0 > /dev/null
    ok "kernel.numa_balancing = 0 (disabled page migration jitter)"
fi

# ==============================================================================
# Memory: Optimize THP defrag (prevent stall during hugepage allocation)
# ==============================================================================
if [ -f /sys/kernel/mm/transparent_hugepage/defrag ]; then
    log "THP defrag -> defer+madvise"
    echo "defer+madvise" > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true
    ok "THP defrag = defer+madvise (prevents allocation stalls)"
fi

# ==============================================================================
# Memory: Allow overcommit for reliable mlock() on large models
# ==============================================================================
CURRENT_OVERCOMMIT=$(cat /proc/sys/vm/overcommit_memory)
log "vm.overcommit_memory: ${CURRENT_OVERCOMMIT} -> 1"
sysctl -w vm.overcommit_memory=1 > /dev/null
ok "vm.overcommit_memory = 1 (mlock() always succeeds)"

# ==============================================================================
# Memory: Dirty page writeback tuning (reduce I/O contention during model load)
# ==============================================================================
sysctl -w vm.dirty_ratio=80 > /dev/null
sysctl -w vm.dirty_background_ratio=5 > /dev/null
ok "vm.dirty_ratio = 80, vm.dirty_background_ratio = 5"

# ==============================================================================
# Memory: Hugepages for reduced TLB misses on large model allocations
# ==============================================================================
CURRENT_HUGEPAGES=$(cat /proc/sys/vm/nr_hugepages)
TARGET_HUGEPAGES=1280  # ~2.5GB of hugepages (1280 * 2MB)
if [ "$CURRENT_HUGEPAGES" -lt "$TARGET_HUGEPAGES" ]; then
    log "vm.nr_hugepages: ${CURRENT_HUGEPAGES} -> ${TARGET_HUGEPAGES}"
    sysctl -w vm.nr_hugepages=${TARGET_HUGEPAGES} > /dev/null
    ok "vm.nr_hugepages = ${TARGET_HUGEPAGES} (~2.5GB hugepages allocated)"
else
    ok "vm.nr_hugepages already >= ${TARGET_HUGEPAGES} (${CURRENT_HUGEPAGES})"
fi

# ==============================================================================
# I/O: Optimize NVMe for model loading
# ==============================================================================
for dev in /sys/block/nvme*; do
    if [ -d "$dev" ]; then
        # Set scheduler to none (NVMe drives handle their own queues)
        echo "none" > "${dev}/queue/scheduler" 2>/dev/null || true
        # Increase read-ahead to 4MB (8192 * 512b) for fast sequential model loads
        echo 8192 > "${dev}/queue/read_ahead_kb" 2>/dev/null || true
    fi
done
ok "NVMe I/O tuned (scheduler=none, read_ahead=4MB)"

# ==============================================================================
# Network: TCP tuning for API latency
# ==============================================================================
sysctl -w net.core.somaxconn=4096 > /dev/null
sysctl -w net.ipv4.tcp_keepalive_time=60 > /dev/null
sysctl -w net.core.rmem_max=16777216 > /dev/null
sysctl -w net.core.wmem_max=16777216 > /dev/null
sysctl -w net.ipv4.tcp_fastopen=3 > /dev/null
ok "TCP tuning applied (somaxconn=4096, fastopen=3, buffers=16MB)"

# ==============================================================================
# PCIe: Disable Active State Power Management (ASPM) for MoE routing latency
# ==============================================================================
# PCIe ASPM saves power by putting the link to sleep, but waking it up adds
# microsecond latency. MoE models rely on instant CPU-to-GPU expert routing.
for dev in /sys/bus/pci/devices/*/power/control; do
    echo "on" > "$dev" 2>/dev/null || true
done
if [ -f /sys/module/pcie_aspm/parameters/policy ]; then
    echo "performance" > /sys/module/pcie_aspm/parameters/policy 2>/dev/null || true
fi
ok "PCIe ASPM disabled (forces links to maximum performance state)"

# ==============================================================================
# CPU: Governor and Energy Bias Hint (EPB)
# ==============================================================================
if command -v cpupower &> /dev/null; then
    CURRENT_GOV=$(cpupower frequency-info -p 2>/dev/null | grep -oP '"[^"]*"' | tr -d '"' || echo "unknown")
    log "CPU governor: ${CURRENT_GOV} -> performance"
    cpupower frequency-set -g performance > /dev/null 2>&1 || warn "Could not set CPU governor"
    # Set Energy Performance Bias to maximum performance (Intel only)
    cpupower set --perf-bias 0 > /dev/null 2>&1 || true
    ok "CPU governor set to performance (EPB=0)"
elif [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "performance" > "$gov" 2>/dev/null || true
    done
    ok "CPU governor set to performance (via sysfs)"
else
    warn "cpupower not found and sysfs not available, skipping CPU governor"
fi

# ==============================================================================
# NVIDIA: Enable persistence mode (avoid cold-start latency)
# ==============================================================================
if command -v nvidia-smi &> /dev/null; then
    CURRENT_PM=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
    if [ "$CURRENT_PM" != "Enabled" ]; then
        log "NVIDIA persistence mode: ${CURRENT_PM} -> Enabled"
        nvidia-smi -pm 1 > /dev/null 2>&1 || warn "Could not enable persistence mode"
        ok "NVIDIA persistence mode enabled (avoids ~100-500ms cold start)"
    else
        ok "NVIDIA persistence mode already enabled"
    fi
else
    warn "nvidia-smi not found, skipping GPU persistence mode"
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Host tuning complete. Changes are NOT persistent.${NC}"
echo -e "${GREEN} To persist, add to /etc/sysctl.d/99-foundry.conf:${NC}"
echo -e "${CYAN}   vm.swappiness = 0${NC}"
echo -e "${CYAN}   vm.overcommit_memory = 1${NC}"
echo -e "${CYAN}   kernel.numa_balancing = 0${NC}"
echo -e "${CYAN}   vm.dirty_ratio = 80${NC}"
echo -e "${CYAN}   vm.dirty_background_ratio = 5${NC}"
echo -e "${CYAN}   vm.nr_hugepages = ${TARGET_HUGEPAGES}${NC}"
echo -e "${CYAN}   net.core.somaxconn = 4096${NC}"
echo -e "${CYAN}   net.core.rmem_max = 16777216${NC}"
echo -e "${CYAN}   net.core.wmem_max = 16777216${NC}"
echo -e "${CYAN}   net.ipv4.tcp_fastopen = 3${NC}"
echo ""
echo -e "${YELLOW} ADVANCED MoE/Blackwell TUNING (Requires GRUB reboot):${NC}"
echo -e "${CYAN} To completely isolate CPU cores for instant MoE expert routing, append this to GRUB_CMDLINE_LINUX:${NC}"
echo -e "${CYAN}   isolcpus=0-15 nohz_full=0-15 rcu_nocbs=0-15 intel_pstate=passive pcie_aspm=off${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
