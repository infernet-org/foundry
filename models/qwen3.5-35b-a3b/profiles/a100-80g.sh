# ==============================================================================
# Foundry Profile: A100 80GB
# ==============================================================================
# Qwen3.5-35B-A3B fits entirely in VRAM even at Q8_0 (~40GB).
# Use higher quality quantization and maximum context.
# Expected: ~80 tok/s generation
# ==============================================================================

PROFILE_CTX_LENGTH=131072
PROFILE_THREADS=32
PROFILE_FIT="off"
PROFILE_FLASH_ATTN="true"
PROFILE_KV_TYPE_K="q8_0"
PROFILE_KV_TYPE_V="q8_0"
PROFILE_NO_MMAP="true"
PROFILE_JINJA="true"
PROFILE_PARALLEL=8
PROFILE_EXTRA_ARGS=""

# Override to use Q8_0 for better quality on datacenter cards
export FOUNDRY_GGUF_FILE="${FOUNDRY_GGUF_FILE:-Qwen3.5-35B-A3B-Q8_0.gguf}"
