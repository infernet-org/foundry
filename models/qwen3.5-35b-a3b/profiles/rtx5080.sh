# ==============================================================================
# Foundry Profile: RTX 5080 (16GB)
# ==============================================================================
# Qwen3.5-35B-A3B Q4_K_M requires partial expert offloading.
# --fit on auto-manages GPU/CPU split. Do NOT set -b/-ub batch flags
# as they consume VRAM that --fit needs for expert layers.
# Expected: ~75 tok/s generation
# ==============================================================================

PROFILE_CTX_LENGTH=32768
PROFILE_THREADS=20
PROFILE_FIT="on"
PROFILE_FLASH_ATTN="true"
PROFILE_KV_TYPE_K="q8_0"
PROFILE_KV_TYPE_V="q8_0"
PROFILE_NO_MMAP="true"
PROFILE_JINJA="true"
PROFILE_PARALLEL=2
PROFILE_EXTRA_ARGS=""
