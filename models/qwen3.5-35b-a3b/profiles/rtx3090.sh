# ==============================================================================
# Foundry Profile: RTX 3090 (24GB)
# ==============================================================================
# Qwen3.5-35B-A3B Q4_K_M (~20GB) fits with partial expert offload.
# PCIe 4.0 and lower memory bandwidth vs 4090.
# Expected: ~55 tok/s generation
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
