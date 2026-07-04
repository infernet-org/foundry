# ==============================================================================
# Foundry Profile: Default (unknown GPU) -- Qwen3.6-35B-A3B-NVFP4 via vLLM
# ==============================================================================
# Conservative settings for any NVFP4-capable GPU with 32 GB+ VRAM
# (Hopper sm_90, Blackwell sm_100/sm_120). The ~22 GB checkpoint does not
# fit on 24 GB cards -- use a GGUF-based foundry model there instead.
# ==============================================================================

PROFILE_CTX_LENGTH=32768        # 32K context, safe baseline
PROFILE_GPU_MEM_UTIL=0.88       # Leave headroom for driver/display
PROFILE_MAX_NUM_SEQS=4          # Conservative concurrency
PROFILE_MAX_BATCHED_TOKENS=2048 # Small prefill chunks, low activation memory
PROFILE_MOE_BACKEND="auto"      # Let vLLM pick per compute capability
PROFILE_MULTIMODAL="false"      # Text-only by default
PROFILE_EXTRA_ARGS=""
