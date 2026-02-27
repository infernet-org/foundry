#!/bin/bash
# ==============================================================================
# Foundry RunPod Serverless Entrypoint
# ==============================================================================
# Handles:
# - GPU detection and profile selection
# - Model download (if not pre-baked into image)
# - Server startup with optimized settings
# - Graceful shutdown on SIGTERM
# ==============================================================================

set -euo pipefail

FOUNDRY_DIR="/opt/foundry"
PROFILES_DIR="${FOUNDRY_DIR}/profiles"
MODELS_DIR="/models"

log() { echo "[foundry] $*"; }

# GPU detection
detect_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "default"
        return
    fi
    
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "")
    
    case "$gpu_name" in
        *"5090"*)       echo "rtx5090" ;;
        *"5080"*)       echo "rtx5080" ;;
        *"4090"*)       echo "rtx4090" ;;
        *"4080"*)       echo "rtx4080" ;;
        *"3090"*)       echo "rtx3090" ;;
        *"A100"*)       echo "a100-80g" ;;
        *"H100"*)       echo "h100" ;;
        *)              echo "default" ;;
    esac
}

# Load profile
load_profile() {
    local profile_name="${1:-default}"
    local profile_file="${PROFILES_DIR}/${profile_name}.sh"
    
    if [ -f "$profile_file" ]; then
        source "$profile_file"
        log "Loaded profile: ${profile_name}"
    else
        log "Profile ${profile_name} not found, using defaults"
    fi
}

# Download model if needed
ensure_model() {
    local gguf_path="${MODELS_DIR}/${FOUNDRY_GGUF_FILE}"
    
    if [ -f "$gguf_path" ]; then
        local size
        size=$(du -h "$gguf_path" | cut -f1)
        log "Model ready: ${gguf_path} (${size})"
        return 0
    fi
    
    log "Downloading model: ${FOUNDRY_GGUF_FILE}"
    log "This may take a few minutes on first cold start..."
    
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 -c "
import os
from huggingface_hub import hf_hub_download
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
hf_hub_download(
    repo_id='${FOUNDRY_GGUF_REPO}',
    filename='${FOUNDRY_GGUF_FILE}',
    local_dir='${MODELS_DIR}',
    token=token
)
"
        log "Model downloaded successfully"
    else
        log "ERROR: huggingface-hub not found"
        exit 1
    fi
}

# Graceful shutdown handler
cleanup() {
    log "Received shutdown signal, stopping server..."
    if [ -n "${SERVER_PID:-}" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    log "Server stopped"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main
main() {
    log "Foundry RunPod Serverless"
    log "=========================="
    
    # Determine and load profile
    local profile
    profile=$(detect_gpu)
    load_profile "$profile"
    
    # Ensure model is available
    ensure_model
    
    # Build server command
    local gguf_path="${MODELS_DIR}/${FOUNDRY_GGUF_FILE}"
    local ctx="${FOUNDRY_CTX_LENGTH:-${PROFILE_CTX_LENGTH:-32768}}"
    local threads="${FOUNDRY_THREADS:-${PROFILE_THREADS:-16}}"
    local parallel="${FOUNDRY_PARALLEL:-${PROFILE_PARALLEL:-2}}"
    
    log "Starting RunPod health check sidecar on port 8081..."
    python3 /opt/foundry/health_check.py &
    
    log "Starting llama-server..."
    log "  Context: ${ctx} tokens"
    log "  Threads: ${threads}"
    log "  Parallel: ${parallel}"
    
    /app/llama-server \
        --model "${gguf_path}" \
        --host "${FOUNDRY_HOST:-0.0.0.0}" \
        --port "${FOUNDRY_PORT:-8080}" \
        --ctx-size "${ctx}" \
        --threads "${threads}" \
        --parallel "${parallel}" \
        --fit on \
        --flash-attn on \
        -ctk q8_0 -ctv q8_0 \
        --no-mmap \
        --jinja \
        ${PROFILE_EXTRA_ARGS:-} &
    
    SERVER_PID=$!
    
    # Wait for server to be healthy
    log "Waiting for server to be ready..."
    for i in {1..60}; do
        if curl -sf "http://localhost:${FOUNDRY_PORT:-8080}/health" > /dev/null 2>&1; then
            log "Server is ready!"
            break
        fi
        sleep 1
    done
    
    # Keep running until shutdown signal
    log "Serving requests on port ${FOUNDRY_PORT:-8080}"
    wait "$SERVER_PID"
}

main "$@"
