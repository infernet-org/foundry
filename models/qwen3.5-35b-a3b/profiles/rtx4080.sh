# ==============================================================================
# Foundry Profile: RTX 4080 (16GB)
# ==============================================================================
# Qwen3.5-35B-A3B Q4_K_M requires significant expert offloading.
# Similar to RTX 5080 but lower memory bandwidth.
# Expected: ~50 tok/s generation
# ==============================================================================

PROFILE_CTX_LENGTH=32768
PROFILE_THREADS=16
PROFILE_FIT="on"
PROFILE_FLASH_ATTN="true"
PROFILE_KV_TYPE_K="q8_0"
PROFILE_KV_TYPE_V="q8_0"
PROFILE_NO_MMAP="true"
PROFILE_JINJA="true"
PROFILE_PARALLEL=2
PROFILE_EXTRA_ARGS=""
