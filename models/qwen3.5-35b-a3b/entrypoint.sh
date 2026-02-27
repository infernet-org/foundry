#!/bin/bash
# ==============================================================================
# Foundry Entrypoint
# ==============================================================================
# 1. Detect GPU and load hardware profile
# 2. Download model if not present
# 3. Launch llama-server with tuned parameters
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
warn() { echo -e "${YELLOW}[foundry]${NC} $*"; }
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

    log "Detected GPU: ${gpu_name}"

    # Map GPU name to profile
    case "$gpu_name" in
        *"5090"*)       echo "rtx5090" ;;
        *"5080"*)       echo "rtx5080" ;;
        *"4090"*)       echo "rtx4090" ;;
        *"4080"*)       echo "rtx4080" ;;
        *"3090"*)       echo "rtx3090" ;;
        *"A100"*"80G"*) echo "a100-80g" ;;
        *"A100"*)       echo "a100-80g" ;;
        *"H100"*)       echo "h100" ;;
        *)
            warn "Unknown GPU '${gpu_name}', using default profile"
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
    source "$profile_file"
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
    log "This is a one-time download (~20GB). Subsequent starts will be instant."
    echo ""

    # Use huggingface-cli if available, fall back to curl
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download \
            "${FOUNDRY_GGUF_REPO}" \
            "${FOUNDRY_GGUF_FILE}" \
            --local-dir "${MODELS_DIR}" \
            --local-dir-use-symlinks False
    else
        err "huggingface-cli not found. Please mount the GGUF at ${gguf_path}"
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

build_command() {
    local gguf_path="${MODELS_DIR}/${FOUNDRY_GGUF_FILE}"

    # Start with base command
    local cmd="llama-server"
    cmd+=" --model ${gguf_path}"
    cmd+=" --host 0.0.0.0"
    cmd+=" --port ${FOUNDRY_PORT:-8080}"

    # Context length (env override > profile > default)
    local ctx="${FOUNDRY_CTX_LENGTH:-${PROFILE_CTX_LENGTH:-32768}}"
    cmd+=" --ctx-size ${ctx}"

    # Thread count (env override > profile > auto)
    local threads="${FOUNDRY_THREADS:-${PROFILE_THREADS:-}}"
    if [ -n "$threads" ]; then
        cmd+=" --threads ${threads}"
    fi

    # Fit mode (MoE expert offloading)
    local fit="${PROFILE_FIT:-on}"
    cmd+=" --fit ${fit}"

    # Flash attention
    if [ "${PROFILE_FLASH_ATTN:-true}" = "true" ]; then
        cmd+=" -fa"
    fi

    # KV cache quantization
    local ctk="${PROFILE_KV_TYPE_K:-q8_0}"
    local ctv="${PROFILE_KV_TYPE_V:-q8_0}"
    cmd+=" -ctk ${ctk} -ctv ${ctv}"

    # Memory mapping
    if [ "${PROFILE_NO_MMAP:-true}" = "true" ]; then
        cmd+=" --no-mmap"
    fi

    # Jinja templates (for tool calling / chat templates)
    if [ "${PROFILE_JINJA:-true}" = "true" ]; then
        cmd+=" --jinja"
    fi

    # Parallel slots for concurrent requests
    local slots="${PROFILE_PARALLEL:-2}"
    cmd+=" --parallel ${slots}"

    # Profile-specific extra args
    if [ -n "${PROFILE_EXTRA_ARGS:-}" ]; then
        cmd+=" ${PROFILE_EXTRA_ARGS}"
    fi

    # User extra args (highest priority override)
    if [ -n "${FOUNDRY_EXTRA_ARGS:-}" ]; then
        cmd+=" ${FOUNDRY_EXTRA_ARGS}"
    fi

    echo "$cmd"
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           Foundry Inference              ║${NC}"
    echo -e "${GREEN}║   github.com/infernet-org/foundry        ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
    echo ""

    log "Model: ${FOUNDRY_MODEL_NAME}"

    # 1. Determine profile
    local profile
    if [ "${FOUNDRY_PROFILE}" = "auto" ]; then
        profile=$(detect_gpu)
    else
        profile="${FOUNDRY_PROFILE}"
    fi

    # 2. Load profile
    load_profile "$profile"

    # 3. Download model if needed
    download_model

    # 4. Build and display launch command
    local cmd
    cmd=$(build_command)

    echo ""
    log "Launch command:"
    echo -e "${CYAN}  ${cmd}${NC}"
    echo ""
    log "OpenAI-compatible API will be available at:"
    echo -e "${GREEN}  http://localhost:${FOUNDRY_PORT:-8080}/v1/chat/completions${NC}"
    echo ""

    # 5. Launch (exec replaces shell process for proper signal handling)
    exec $cmd
}

main "$@"
