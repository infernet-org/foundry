# ==============================================================================
# Foundry Profile: RTX 4090 (24GB)
# ==============================================================================
# Qwen3.5-35B-A3B Q4_K_M (~20GB) fits in VRAM with partial expert offload.
# --fit on handles the split automatically. Room for moderate context.
# Expected: ~70 tok/s generation
# ==============================================================================

PROFILE_CTX_LENGTH=65536
PROFILE_THREADS=16
PROFILE_FIT="on"
PROFILE_FLASH_ATTN="true"
PROFILE_KV_TYPE_K="q8_0"
PROFILE_KV_TYPE_V="q8_0"
PROFILE_NO_MMAP="true"
PROFILE_JINJA="true"
PROFILE_PARALLEL=2
PROFILE_EXTRA_ARGS=""
