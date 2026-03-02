#!/bin/bash
# ==============================================================================
# Foundry Entrypoint (shared across all models)
# ==============================================================================
# 1. Detect GPU and load hardware profile
# 2. Download model if not present
# 3. Apply architecture-aware tuning (MoE vs Dense)
# 4. Launch llama-server with tuned parameters
#
# Model identity is set via Dockerfile ENV vars:
#   FOUNDRY_MODEL_NAME   -- display name (e.g. "Qwen3.5-35B-A3B")
#   FOUNDRY_GGUF_REPO    -- HuggingFace repo (e.g. "unsloth/Qwen3.5-35B-A3B-GGUF")
#   FOUNDRY_GGUF_FILE    -- GGUF filename
#   FOUNDRY_ARCH         -- architecture type: "moe" or "dense"
#
# Architecture-specific flags are applied automatically based on FOUNDRY_ARCH:
#
#   MoE (e.g. Qwen3.5-35B-A3B):
#     --fit on         Expert offloading: spill inactive experts to CPU
#
#   Dense (e.g. Hermes-4.3-36B):
#     (no --fit)       No experts to offload
#
# Model-specific quirks (e.g. --swa-full for hybrid attention, --cache-ram 0
# for recurrent architectures) belong in PROFILE_EXTRA_ARGS, NOT in the arch
# tier -- they are not universal to the architecture class.
#
# Hardware-specific tuning (context, threads, KV quant, slots) is set by
# per-GPU profiles in /opt/foundry/profiles/*.sh.
# ==============================================================================

set -euo pipefail

FOUNDRY_DIR="/opt/foundry"
PROFILES_DIR="${FOUNDRY_DIR}/profiles"
MODELS_DIR="/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${CYAN}[foundry]${NC} $*"; }
warn() { echo -e "${YELLOW}[foundry]${NC} $*" >&2; }
err()  { echo -e "${RED}[foundry]${NC} $*" >&2; }

# ==============================================================================
# GPU Detection
# ==============================================================================

detect_gpu() {
    local gpu_name
    if ! command -v nvidia-smi &> /dev/null; then
        warn "nvidia-smi not found, using default profile"
        echo "default"
        return
    fi

    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)

    if [ -z "$gpu_name" ]; then
        warn "Could not detect GPU, using default profile"
        echo "default"
        return
    fi

    # Log to stderr so it doesn't interfere with the captured profile name
    log "Detected GPU: ${gpu_name}" >&2

    # Map GPU name to profile
    case "$gpu_name" in
        *"5090"*)       echo "rtx5090" ;;
        *)
            warn "Unknown or unsupported GPU '${gpu_name}', using default profile"
            echo "default"
            ;;
    esac
}

# ==============================================================================
# Profile Loading
# ==============================================================================

load_profile() {
    local profile_name="$1"
    local profile_file="${PROFILES_DIR}/${profile_name}.sh"

    if [ ! -f "$profile_file" ]; then
        warn "Profile '${profile_name}' not found, falling back to default"
        profile_file="${PROFILES_DIR}/default.sh"
    fi

    if [ ! -f "$profile_file" ]; then
        err "No default profile found at ${profile_file}"
        exit 1
    fi

    log "Loading profile: ${profile_name}"
    # shellcheck source=profiles/default.sh
    source "$profile_file"
}

# ==============================================================================
# Pre-flight Host Check
# ==============================================================================
# Checks host kernel parameters visible via /proc/sys/ and warns about
# suboptimal settings that affect inference performance. Does NOT block startup.
#
# These parameters are set by: sudo ./scripts/host-setup.sh

preflight_check() {
    local warnings=0

    # Helper: check a sysctl value and warn if it doesn't match
    check_sysctl() {
        local path="$1" expected="$2" label="$3" fix="$4"
        local actual
        if [ -f "$path" ]; then
            actual=$(cat "$path" 2>/dev/null)
            if [ "$actual" != "$expected" ]; then
                warn "  $label = $actual (expected: $expected) -- $fix"
                warnings=$((warnings + 1))
            fi
        fi
    }

    log "Checking host kernel parameters..."

    # TCP congestion control (BBR reduces streaming latency)
    check_sysctl /proc/sys/net/ipv4/tcp_congestion_control "bbr" \
        "tcp_congestion_control" "BBR reduces token streaming latency"

    # NUMA balancing (causes random latency spikes during decode)
    check_sysctl /proc/sys/kernel/numa_balancing "0" \
        "numa_balancing" "page migration causes decode latency jitter"

    # Swappiness (model weights should stay in RAM)
    local swappiness
    swappiness=$(cat /proc/sys/vm/swappiness 2>/dev/null)
    if [ -n "$swappiness" ] && [ "$swappiness" -gt 10 ]; then
        warn "  vm.swappiness = $swappiness (expected: 0-10) -- high swappiness may evict model weights"
        warnings=$((warnings + 1))
    fi

    # GPU persistence mode
    if command -v nvidia-smi &> /dev/null; then
        local pm
        pm=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
        if [ "$pm" = "Disabled" ]; then
            warn "  GPU persistence mode = Disabled -- adds 100-500ms cold start latency"
            warnings=$((warnings + 1))
        fi
    fi

    if [ "$warnings" -gt 0 ]; then
        warn ""
        warn "  $warnings performance issue(s) detected. Run on the host:"
        warn "    sudo ./scripts/host-setup.sh"
        warn ""
    else
        log "Host kernel parameters: OK"
    fi
}

# ==============================================================================
# Model Download
# ==============================================================================

download_model() {
    local gguf_path="${MODELS_DIR}/${FOUNDRY_GGUF_FILE}"

    if [ -f "$gguf_path" ]; then
        local size
        size=$(du -h "$gguf_path" | cut -f1)
        log "Model found: ${gguf_path} (${size})"
        return 0
    fi

    log "Model not found at ${gguf_path}"
    log "Downloading ${FOUNDRY_GGUF_FILE} from ${FOUNDRY_GGUF_REPO}..."
    log "This is a one-time download. Subsequent starts will be instant."
    echo ""

    # Use python3 huggingface_hub to download (huggingface-cli may not be on PATH)
    # Variables are passed via environment to avoid shell injection in inline Python
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        FOUNDRY_GGUF_REPO="${FOUNDRY_GGUF_REPO}" \
        FOUNDRY_GGUF_FILE="${FOUNDRY_GGUF_FILE}" \
        FOUNDRY_MODELS_DIR="${MODELS_DIR}" \
        python3 -c "
import os
from huggingface_hub import hf_hub_download
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
hf_hub_download(
    repo_id=os.environ['FOUNDRY_GGUF_REPO'],
    filename=os.environ['FOUNDRY_GGUF_FILE'],
    local_dir=os.environ['FOUNDRY_MODELS_DIR'],
    token=token
)
"
    else
        err "huggingface-hub not found. Please mount the GGUF at ${gguf_path}"
        err "Or install huggingface-hub: pip install huggingface-hub"
        exit 1
    fi

    if [ ! -f "$gguf_path" ]; then
        err "Download failed: ${gguf_path} not found after download"
        exit 1
    fi

    local size
    size=$(du -h "$gguf_path" | cut -f1)
    log "Download complete: ${gguf_path} (${size})"
}

# ==============================================================================
# Build Launch Command
# ==============================================================================
# Flags are layered in three tiers:
#   1. Architecture defaults (FOUNDRY_ARCH)  -- systematic, model-class level
#   2. Hardware profile (PROFILE_*)          -- per-GPU tuning knobs
#   3. User overrides (FOUNDRY_EXTRA_ARGS)   -- escape hatch, highest priority

build_command() {
    local gguf_path="${MODELS_DIR}/${FOUNDRY_GGUF_FILE}"
    local arch="${FOUNDRY_ARCH:-dense}"

    # Use a bash array to safely handle arguments with spaces
    local -a cmd=("/app/llama-server")
    cmd+=("--model" "${gguf_path}")
    cmd+=("--host" "0.0.0.0")
    cmd+=("--port" "${FOUNDRY_PORT:-8080}")

    # --- Tier 1: Architecture-specific flags ----------------------------------
    # These are determined by the model class, not by the GPU or user preference.

    if [ "$arch" = "moe" ]; then
        # MoE: enable expert offloading (spill inactive experts to CPU when VRAM
        # is tight). On high-VRAM GPUs --fit keeps everything on GPU automatically.
        cmd+=("--fit" "on")
    fi
    # Dense models: no --fit (no experts to offload).
    # Model-specific flags (--swa-full, --cache-ram) go in PROFILE_EXTRA_ARGS.

    # --- Tier 2: Hardware profile tuning --------------------------------------
    # These come from the sourced profile .sh file and tune for the specific GPU.

    # Context length (env override > profile > default)
    local ctx="${FOUNDRY_CTX_LENGTH:-${PROFILE_CTX_LENGTH:-32768}}"
    cmd+=("--ctx-size" "${ctx}")

    # Thread count (env override > profile > auto)
    local threads="${FOUNDRY_THREADS:-${PROFILE_THREADS:-}}"
    if [ -n "$threads" ]; then
        cmd+=("--threads" "${threads}")
    fi

    # Batch thread count (can be higher than decode threads for prompt processing)
    local threads_batch="${PROFILE_THREADS_BATCH:-${threads}}"
    if [ -n "$threads_batch" ]; then
        cmd+=("--threads-batch" "${threads_batch}")
    fi

    # Flash attention (new llama.cpp requires explicit on/off/auto value)
    local fa="${PROFILE_FLASH_ATTN:-on}"
    cmd+=("--flash-attn" "${fa}")

    # KV cache quantization
    local ctk="${PROFILE_KV_TYPE_K:-q8_0}"
    local ctv="${PROFILE_KV_TYPE_V:-q8_0}"
    cmd+=("-ctk" "${ctk}" "-ctv" "${ctv}")

    # Memory mapping
    if [ "${PROFILE_NO_MMAP:-true}" = "true" ]; then
        cmd+=("--no-mmap")
    fi

    # Jinja templates (for tool calling / chat templates)
    if [ "${PROFILE_JINJA:-true}" = "true" ]; then
        cmd+=("--jinja")
    fi

    # Parallel slots for concurrent requests
    local slots="${PROFILE_PARALLEL:-2}"
    cmd+=("--parallel" "${slots}")

    # Thread priority for reduced scheduling latency
    local prio="${PROFILE_PRIO:-0}"
    if [ "$prio" != "0" ]; then
        cmd+=("--prio" "${prio}")
    fi

    # Strict CPU placement for cache locality
    if [ "${PROFILE_CPU_STRICT:-0}" = "1" ]; then
        cmd+=("--cpu-strict" "1")
    fi

    # KV cache reuse for multi-turn chat (prefix sharing via KV shifting)
    local cache_reuse="${PROFILE_CACHE_REUSE:-0}"
    if [ "$cache_reuse" != "0" ]; then
        cmd+=("--cache-reuse" "${cache_reuse}")
    fi

    # Disable web UI for headless server deployments
    if [ "${PROFILE_NO_WEBUI:-false}" = "true" ]; then
        cmd+=("--no-webui")
    fi

    # Prometheus-compatible metrics endpoint
    if [ "${PROFILE_METRICS:-false}" = "true" ]; then
        cmd+=("--metrics")
    fi

    # Profile-specific extra args (split on spaces intentionally)
    if [ -n "${PROFILE_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        cmd+=(${PROFILE_EXTRA_ARGS})
    fi

    # --- Tier 3: User overrides -----------------------------------------------
    if [ -n "${FOUNDRY_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        cmd+=(${FOUNDRY_EXTRA_ARGS})
    fi

    # Store the array globally so main() can exec it safely
    FOUNDRY_CMD=("${cmd[@]}")
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║            Foundry Inference               ║${NC}"
    echo -e "${GREEN}║   github.com/infernet-org/foundry          ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
    echo ""

    log "Model: ${FOUNDRY_MODEL_NAME}"
    log "Architecture: ${FOUNDRY_ARCH:-dense}"

    # 1. Determine profile
    local profile
    if [ "${FOUNDRY_PROFILE}" = "auto" ]; then
        profile=$(detect_gpu)
    else
        profile="${FOUNDRY_PROFILE}"
    fi

    # 2. Load profile
    load_profile "$profile"

    # 3. Pre-flight host check (warns about suboptimal kernel params)
    preflight_check

    # 4. Download model if needed
    download_model

    # 5. Build launch command (sets FOUNDRY_CMD array directly, no subshell)
    build_command

    echo ""
    log "Launch command:"
    echo -e "${CYAN}  ${FOUNDRY_CMD[*]}${NC}"
    echo ""
    log "OpenAI-compatible API will be available at:"
    echo -e "${GREEN}  http://localhost:${FOUNDRY_PORT:-8080}/v1/chat/completions${NC}"
    echo ""

    # 6. Launch (exec replaces shell process for proper signal handling)
    # Use the array form to avoid word-splitting issues
    exec "${FOUNDRY_CMD[@]}"
}

main "$@"
