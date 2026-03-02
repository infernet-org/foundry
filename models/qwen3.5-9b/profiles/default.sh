# ==============================================================================
# Foundry Profile: Default (8GB+ VRAM)
# ==============================================================================
# Qwen3.5-9B UD-Q4_K_XL (~5.66GB)
#
# Conservative profile for GPUs with 8-16GB VRAM.
# At only 5.66GB model weight, this is the lightest model in the lineup
# and runs comfortably on 8GB cards with reduced context.
# ==============================================================================

PROFILE_CTX_LENGTH=32768        # 32K context -- safe for 8GB+ cards
PROFILE_THREADS=8               # Conservative thread count
PROFILE_THREADS_BATCH=8
PROFILE_FLASH_ATTN="on"
PROFILE_KV_TYPE_K="q4_0"        # Aggressive KV quantization to save VRAM
PROFILE_KV_TYPE_V="q4_0"
PROFILE_NO_MMAP="true"
PROFILE_JINJA="true"            # Tool calling support
PROFILE_PARALLEL=2              # 2 slots for smaller GPUs
PROFILE_PRIO=0                  # Normal priority (conservative)
PROFILE_CPU_STRICT=0
PROFILE_CACHE_REUSE=0           # Disabled: hybrid recurrent arch re-processes anyway
PROFILE_NO_WEBUI="false"        # Keep web UI for exploration
PROFILE_METRICS="false"
PROFILE_EXTRA_ARGS="--swa-full --cache-ram 0 --reasoning-format none"
