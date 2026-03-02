# ==============================================================================
# Foundry Profile: Default (16GB+ VRAM)
# ==============================================================================
# Qwen3-Coder-30B-A3B-Instruct UD-Q4_K_XL (~17.7GB)
#
# Conservative profile for GPUs with 16-24GB VRAM.
# At 17.7GB model weight, this is the lightest model in the lineup and
# has the most headroom on 16GB cards with MoE expert offloading.
# ==============================================================================

PROFILE_CTX_LENGTH=32768        # 32K context -- safe for 16GB+ cards
PROFILE_THREADS=8               # Conservative thread count
PROFILE_THREADS_BATCH=8
PROFILE_FLASH_ATTN="on"
PROFILE_KV_TYPE_K="q4_0"        # Aggressive KV quantization to save VRAM
PROFILE_KV_TYPE_V="q4_0"
PROFILE_NO_MMAP="true"
PROFILE_JINJA="true"            # Tool calling support
PROFILE_PARALLEL=2              # 2 slots for smaller GPUs
PROFILE_PRIO=2
PROFILE_CPU_STRICT=0
PROFILE_CACHE_REUSE=256
PROFILE_NO_WEBUI="true"
PROFILE_METRICS="true"
PROFILE_EXTRA_ARGS="--mlock"
