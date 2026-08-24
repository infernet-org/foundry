# shellcheck shell=bash disable=SC2034
# ==============================================================================
# Foundry Profile: Default (unknown GPU) -- Qwen3.8-27B-NVFP4 via vLLM
# ==============================================================================
# Conservative settings for any NVFP4-capable GPU with 28 GB+ VRAM
# (Hopper sm_90, Blackwell sm_100/sm_120). The ~22 GB checkpoint does not
# fit comfortably on 24 GB cards -- use a GGUF build of this model with
# llama.cpp there instead.
#
# Unlike the RTX 5090 profile this does NOT force --enforce-eager: cards in
# this class (H100/H200/Blackwell Pro) have enough spare VRAM for CUDA graph
# capture alongside the ~24.6 GiB of NVFP4 weights.
# ==============================================================================

PROFILE_CTX_LENGTH=32768        # 32K context, safe baseline
PROFILE_GPU_MEM_UTIL=0.88       # Leave headroom for driver/display
PROFILE_MAX_NUM_SEQS=4          # Conservative concurrency
PROFILE_MAX_BATCHED_TOKENS=2048 # Small prefill chunks, low activation memory
PROFILE_KV_CACHE_DTYPE=fp8      # ~2x KV capacity vs bf16 on hybrid attention
PROFILE_MOE_BACKEND="auto"      # Dense model: let vLLM pick kernels
PROFILE_MULTIMODAL="false"      # Text-only by default
PROFILE_EXTRA_ARGS=""
