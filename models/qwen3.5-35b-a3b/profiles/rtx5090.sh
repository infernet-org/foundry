# ==============================================================================
# Foundry Profile: RTX 5090 (32GB)
# ==============================================================================
# Qwen3.5-35B-A3B Q4_K_M fits entirely in VRAM (~20GB).
# No expert offloading needed. Maximum context window.
# Expected: ~160 tok/s generation
# ==============================================================================

PROFILE_CTX_LENGTH=131072
PROFILE_THREADS=20
PROFILE_FIT="on"
PROFILE_FLASH_ATTN="true"
PROFILE_KV_TYPE_K="q8_0"
PROFILE_KV_TYPE_V="q8_0"
PROFILE_NO_MMAP="true"
PROFILE_JINJA="true"
PROFILE_PARALLEL=4
PROFILE_EXTRA_ARGS=""
