#!/bin/bash
# ==============================================================================
# Foundry Entrypoint: vLLM backend (Qwen3.8-27B-NVFP4)
# ==============================================================================
# vLLM variant of the shared llama.cpp entrypoint. Same flow:
#   1. Detect GPU and load hardware profile
#   2. Download model if not present
#   3. Launch vllm serve with tuned parameters
#
# Model identity is set via Dockerfile ENV vars:
#   FOUNDRY_MODEL_NAME  -- display name (e.g. "Qwen3.8-27B-NVFP4")
#   FOUNDRY_HF_REPO     -- HuggingFace repo (e.g. "unsloth/Qwen3.8-27B-NVFP4")
#   FOUNDRY_MODEL_DIR   -- directory name under /models for the snapshot
#
# Architecture: Qwen3_5ForConditionalGeneration -- dense 27B on a hybrid
# backbone: 16 of 64 layers run full attention (full_attention_interval=4),
# the other 48 run linear attention with constant recurrent state, plus a
# built-in MTP draft head and a BF16 vision tower.
#
# NVFP4 requires compute capability >= 9.0 (Hopper) or 10.x/12.x (Blackwell).
# On consumer Blackwell (sm_120) vLLM selects a native CUTLASS/FlashInfer NVFP4
# kernel for both uniform-W4A4 and mixed FP8+NVFP4 checkpoints.
#
# Quantization is NOT forced: vLLM auto-detects from checkpoint config, which
# is the verified path for the Unsloth mixed FP8+NVFP4 repack. Set
# FOUNDRY_QUANTIZATION (e.g. "modelopt") to force a specific loader.
# ==============================================================================

set -euo pipefail

FOUNDRY_DIR="/opt/foundry"
PROFILES_DIR="${FOUNDRY_DIR}/profiles"
MODELS_DIR="/models"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

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

    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || true)

    if [ -z "$gpu_name" ]; then
        warn "Could not detect GPU, using default profile"
        echo "default"
        return
    fi

    log "Detected GPU: ${gpu_name}" >&2

    case "$gpu_name" in
        *"5090"*)       echo "rtx5090" ;;
        *)
            warn "Unknown or unsupported GPU '${gpu_name}', using default profile"
            echo "default"
            ;;
    esac
}

# ==============================================================================
# NVFP4 Capability Check
# ==============================================================================
# NVFP4 checkpoints need Hopper (sm_90) or Blackwell (sm_100/sm_120).
# Fail fast with a clear message instead of a kernel-selection traceback.

check_compute_capability() {
    if ! command -v nvidia-smi &> /dev/null; then
        return 0  # container without nvidia-smi in PATH; vLLM will check
    fi
    local cap
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || true)
    if [ -z "$cap" ]; then
        return 0
    fi
    local major="${cap%%.*}"
    if [ "$major" -lt 9 ]; then
        err "GPU compute capability ${cap} does not support NVFP4."
        err "This model requires Hopper (sm_90) or Blackwell (RTX 50xx, sm_120)."
        err "Run a GGUF build of this model with llama.cpp on this GPU instead."
        exit 1
    fi
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
# Model Download
# ==============================================================================
# Full snapshot (multi-file safetensors), not a single GGUF.

download_model() {
    local model_path="${MODELS_DIR}/${FOUNDRY_MODEL_DIR}"
    local marker="${model_path}/.foundry_download_complete"

    # config.json alone is not proof of a complete snapshot (it lands first;
    # an interrupted pull leaves truncated shards). Trust only the marker;
    # otherwise run snapshot_download, which resumes/no-ops incrementally.
    if [ -f "$marker" ]; then
        local size
        size=$(du -sh "$model_path" | cut -f1)
        log "Model found: ${model_path} (${size})"
        return 0
    fi

    if [ -f "${model_path}/config.json" ]; then
        log "Model dir exists but has no completion marker -- verifying/resuming download..."
    else
        log "Model not found at ${model_path}"
        log "Downloading ${FOUNDRY_HF_REPO} (~22 GB)..."
        log "This is a one-time download. Subsequent starts will be instant."
    fi
    echo ""

    FOUNDRY_HF_REPO="${FOUNDRY_HF_REPO}" \
    FOUNDRY_MODEL_PATH="${model_path}" \
    python3 -c "
import os
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
snapshot_download(
    repo_id=os.environ['FOUNDRY_HF_REPO'],
    local_dir=os.environ['FOUNDRY_MODEL_PATH'],
    token=token,
)
"

    if [ ! -f "${model_path}/config.json" ]; then
        err "Download failed: ${model_path}/config.json not found after download"
        exit 1
    fi

    touch "$marker"
    local size
    size=$(du -sh "$model_path" | cut -f1)
    log "Download complete: ${model_path} (${size})"
}

# ==============================================================================
# Build Launch Command
# ==============================================================================
# Flags are layered in three tiers (same as the llama.cpp entrypoint):
#   1. Model defaults                          -- parsers, tool calling, name
#   2. Hardware profile (PROFILE_*)            -- per-GPU tuning knobs
#   3. User overrides (FOUNDRY_EXTRA_ARGS)     -- escape hatch, highest priority

build_command() {
    local model_path="${MODELS_DIR}/${FOUNDRY_MODEL_DIR}"

    local -a cmd=("vllm" "serve" "${model_path}")
    cmd+=("--host" "0.0.0.0")
    cmd+=("--port" "${FOUNDRY_PORT:-8080}")

    # --- Tier 1: Model defaults ------------------------------------------------
    cmd+=("--served-model-name" "qwen3.8-27b-nvfp4")

    # Quantization: auto-detect by default (verified path for this checkpoint).
    # FOUNDRY_QUANTIZATION forces an explicit loader when debugging.
    if [ -n "${FOUNDRY_QUANTIZATION:-}" ]; then
        cmd+=("--quantization" "${FOUNDRY_QUANTIZATION}")
    fi

    cmd+=("--reasoning-parser" "qwen3")
    # Tool calling on by default -- foundry serves coding agents, and agent
    # frameworks (mini-SWE-agent, OpenHands, ...) send tool_choice=auto, which
    # vLLM rejects unless a parser is enabled. Qwen3.8 emits qwen3_coder-style
    # tool_call blocks per the official vLLM serving recipe.
    cmd+=("--enable-auto-tool-choice")
    cmd+=("--tool-call-parser" "qwen3_coder")

    # --- Tier 2: Hardware profile tuning ----------------------------------------

    # Context length (env override > profile > default)
    local ctx="${FOUNDRY_CTX_LENGTH:-${PROFILE_CTX_LENGTH:-32768}}"
    cmd+=("--max-model-len" "${ctx}")

    # Fraction of VRAM vLLM may claim (weights + KV cache + buffers)
    cmd+=("--gpu-memory-utilization" "${PROFILE_GPU_MEM_UTIL:-0.90}")

    # Concurrent sequence cap
    cmd+=("--max-num-seqs" "${PROFILE_MAX_NUM_SEQS:-8}")

    # Chunked-prefill batch size; bounds activation memory during profiling
    cmd+=("--max-num-batched-tokens" "${PROFILE_MAX_BATCHED_TOKENS:-4096}")

    # KV cache dtype: fp8 doubles KV capacity vs bf16 on this hybrid model
    # (only full-attention layers carry KV). Empty = vLLM default.
    if [ -n "${PROFILE_KV_CACHE_DTYPE:-}" ]; then
        cmd+=("--kv-cache-dtype" "${PROFILE_KV_CACHE_DTYPE}")
    fi

    # MoE backend knob kept for symmetry with other foundry models; Qwen3.8-27B
    # is dense, so it is normally unset/"auto".
    local moe_backend="${PROFILE_MOE_BACKEND:-auto}"
    if [ "$moe_backend" != "auto" ]; then
        cmd+=("--moe-backend" "${moe_backend}")
    fi

    # Multimodal input: the vision tower is unquantized BF16; enabling it
    # costs VRAM for encoder profiling. Off by default on 32 GB GPUs.
    if [ "${PROFILE_MULTIMODAL:-false}" != "true" ]; then
        cmd+=("--limit-mm-per-prompt.image" "0")
        cmd+=("--limit-mm-per-prompt.video" "0")
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

    # MTP speculative decoding is a model-level requirement pair: the MTP head
    # rejects the default mamba cache mode "all" on this hybrid linear-attention
    # backbone. If any tier enabled MTP without picking a cache mode, add the
    # required one.
    local joined="${cmd[*]}"
    if [[ "$joined" == *'"method":"mtp"'* && "$joined" != *"--mamba-cache-mode"* ]]; then
        cmd+=("--mamba-cache-mode" "align")
    fi

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
    log "Backend: vLLM (NVFP4)"

    # 1. NVFP4 needs Hopper/Blackwell -- fail fast otherwise
    check_compute_capability

    # 2. Determine profile
    local profile
    if [ "${FOUNDRY_PROFILE}" = "auto" ]; then
        profile=$(detect_gpu)
    else
        profile="${FOUNDRY_PROFILE}"
    fi

    # 3. Load profile
    load_profile "$profile"

    # 4. Download model if needed
    download_model

    # 5. Build launch command
    build_command

    echo ""
    log "Launch command:"
    echo -e "${CYAN}  ${FOUNDRY_CMD[*]}${NC}"
    echo ""
    log "OpenAI-compatible API will be available at:"
    echo -e "${GREEN}  http://localhost:${FOUNDRY_PORT:-8080}/v1/chat/completions${NC}"
    echo ""
    log "Note: first startup takes 2-4 minutes (weight loading + graph capture)."

    # 6. Launch (exec replaces shell process for proper signal handling)
    exec "${FOUNDRY_CMD[@]}"
}

main "$@"
