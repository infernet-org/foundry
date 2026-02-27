# ==============================================================================
# Foundry Profile: Default (unknown GPU)
# ==============================================================================
# Conservative settings that should work on any 16GB+ NVIDIA GPU.
# Uses --fit on for automatic GPU/CPU memory management.
# Expected: varies by GPU
# ==============================================================================

PROFILE_CTX_LENGTH=32768
PROFILE_THREADS=8
PROFILE_FIT="on"
PROFILE_FLASH_ATTN="true"
PROFILE_KV_TYPE_K="q8_0"
PROFILE_KV_TYPE_V="q8_0"
PROFILE_NO_MMAP="true"
PROFILE_JINJA="true"
PROFILE_PARALLEL=2
PROFILE_EXTRA_ARGS=""
