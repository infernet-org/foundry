# ==============================================================================
# Foundry Profile: RTX 5090 (32GB)
# ==============================================================================
# Qwen3-Coder-30B-A3B-Instruct UD-Q4_K_XL (~17.7GB)
#
# Architecture: Qwen3 MoE (standard transformer, NOT hybrid DeltaNet)
#   - Standard MoE with full KV cache (no recurrent layers)
#   - 128 experts per MoE layer, top-8 active per token (~3B active)
#   - Optimized for code generation and tool calling
#
# Why --parallel 3 (not 4):
#   With 3 slots, --fit on allocates 64K context per slot (vs 48K with 4 slots).
#   This is 33% more context per agent with identical aggregate throughput:
#     3 slots: 275 tok/s single | 497 tok/s agg | 168 tok/s each | 64K/slot
#     4 slots: 274 tok/s single | 495 tok/s agg | 124 tok/s each | 48K/slot
#   The 3rd slot queues only when 3+ requests are in-flight simultaneously,
#   and per-agent speed under load is 35% faster (168 vs 124 tok/s).
#
# VRAM budget (32,607 MiB total):
#   Model weights:    ~17.7 GB
#   KV cache (192K):  ~9.8 GB (3 slots x 64K, q8_0)
#   Compute buffers:  ~2.4 GB
#   Free headroom:    ~2.7 GB
#
# Key differences from Qwen3.5-35B-A3B profile:
#   - No --swa-full (not a hybrid model, no sliding window attention)
#   - No --cache-ram 0 (standard KV cache, prompt caching works normally)
#   - --parallel 3 (vs 4 for Qwen3.5, which has smaller KV due to recurrent layers)
#   - cache-reuse enabled (effective for coding workflows with repeated context)
#
# Benchmarked on RTX 5090 (2026-03-02, native sm_120a, BLACKWELL_NATIVE_FP4=1):
#   Single-stream decode:  ~275 tok/s  (memory-bandwidth-bound)
#   3-concurrent aggregate: ~497 tok/s (+81% via MoE expert batching)
#   3-concurrent per-slot:  ~168 tok/s each
#   Prompt processing:    ~345-1,038 tok/s (varies with batch position)
# ==============================================================================

PROFILE_CTX_LENGTH=196608       # 192K total -- --fit on allocates 64K per slot with 3 slots
PROFILE_THREADS=16              # Physical cores (avoid hyperthreads for decode)
PROFILE_THREADS_BATCH=20        # Higher thread count for prompt processing
PROFILE_FLASH_ATTN="on"         # Flash attention for long context perf
PROFILE_KV_TYPE_K="q8_0"        # KV cache key quantization
PROFILE_KV_TYPE_V="q8_0"        # KV cache value quantization
PROFILE_NO_MMAP="true"          # Avoid page faults, load model into RAM
PROFILE_JINJA="true"            # Chat template / tool calling support
PROFILE_PARALLEL=3              # 3 slots: 64K/slot, 497 tok/s agg, 168 tok/s each
                                # (see "Why --parallel 3" above)
PROFILE_PRIO=2                  # High thread priority for reduced scheduling latency
PROFILE_CPU_STRICT=1            # Strict CPU placement for cache locality
PROFILE_CACHE_REUSE=256         # KV cache reuse for multi-turn coding sessions
PROFILE_NO_WEBUI="true"         # Headless: no web UI, reduce attack surface
PROFILE_METRICS="true"          # Prometheus-compatible /metrics endpoint
# --mlock: pin model in RAM; -b/-ub 4096: large batch for fast prompt encode
PROFILE_EXTRA_ARGS="--mlock -b 4096 -ub 4096"
