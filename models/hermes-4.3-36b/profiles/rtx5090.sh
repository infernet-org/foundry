# ==============================================================================
# Foundry Profile: RTX 5090 (32GB)
# ==============================================================================
# Hermes-4.3-36B (Dense 36B) Q4_K_M (~21.8GB)
#
# Architecture: 36B Dense, GQA (80 Q, 8 KV)
# VRAM budget (32,607 MiB total):
#   Model weights:    ~21.8 GB
#   KV cache (32K):   ~4.2 GB (split across 4 slots)
#   Compute buffers:  ~2.5 GB
#   Free headroom:    ~4.1 GB
#
# Dense models are extremely memory bandwidth heavy.
# Benchmarked on RTX 5090 (2026-03-01):
#   Pure decode speed: ~64 tok/s (single-stream)
#   4-concurrent agg:  ~170 tok/s
# ==============================================================================

PROFILE_CTX_LENGTH=32768        # 32K context -- safe for 32GB VRAM with 4 slots
PROFILE_THREADS=16              # Physical cores
PROFILE_THREADS_BATCH=20        
PROFILE_FLASH_ATTN="on"         
PROFILE_KV_TYPE_K="q8_0"        # q8_0 empirically faster than q4_0 for dense batched decode
PROFILE_KV_TYPE_V="q8_0"        
PROFILE_NO_MMAP="true"          
PROFILE_JINJA="true"            
PROFILE_PARALLEL=4              # 4 slots yields ~170 tok/s aggregate (sweet spot)
PROFILE_PRIO=2                  
PROFILE_CPU_STRICT=1            
PROFILE_CACHE_REUSE=256         # Important for dense models reasoning/tool use
PROFILE_NO_WEBUI="true"         
PROFILE_METRICS="true"          
PROFILE_EXTRA_ARGS="--mlock -b 4096 -ub 4096"

