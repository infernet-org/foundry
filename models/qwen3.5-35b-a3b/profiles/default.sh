# ==============================================================================
# Foundry Profile: Default (unknown GPU)
# ==============================================================================
# Conservative settings that should work on any 16GB+ NVIDIA GPU.
# Uses --fit on for automatic GPU/CPU memory management.
# Single request slot for maximum throughput.
# ==============================================================================

PROFILE_CTX_LENGTH=16384        # 16K context, safe for any GPU
PROFILE_THREADS=8               # Conservative thread count
PROFILE_THREADS_BATCH=8         # Match decode threads for safety
PROFILE_FLASH_ATTN="on"         # Flash attention
PROFILE_KV_TYPE_K="q4_0"        # Aggressive KV quant for unknown VRAM
PROFILE_KV_TYPE_V="q4_0"        # Saves memory on constrained GPUs
PROFILE_NO_MMAP="true"          # Avoid page faults
PROFILE_JINJA="true"            # Chat template support
PROFILE_PARALLEL=1              # Single slot = safest, fastest default
PROFILE_PRIO=0                  # Normal priority (conservative)
PROFILE_CPU_STRICT=0            # No strict placement (conservative)
PROFILE_CACHE_REUSE=256         # KV cache reuse for multi-turn chat
PROFILE_NO_WEBUI="false"        # Keep web UI available for exploration
PROFILE_METRICS="false"         # Metrics disabled by default
PROFILE_EXTRA_ARGS=""
