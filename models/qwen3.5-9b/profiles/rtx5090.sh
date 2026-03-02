# ==============================================================================
# Foundry Profile: RTX 5090 (32GB)
# ==============================================================================
# Qwen3.5-9B UD-Q4_K_XL (~5.66GB) -- Dense model, Qwen3.5 generation
#
# Architecture: Hybrid Gated DeltaNet + Dense FFN (NOT MoE)
#   - 32 layers: 24 Gated DeltaNet (recurrent) + 8 full attention (GQA 16:4)
#   - All 9B parameters active per token (dense, compute-bound)
#   - Vision-language capable (multimodal)
#   - Thinking mode by default (reasoning_content field)
#
# Why this model replaces Qwen3.5-35B-A3B:
#   Qwen3.5-9B is a newer generation (Qwen3.5) that dramatically outperforms
#   the older 35B-A3B on every benchmark: agent tasks (+37 TAU2-Bench),
#   math (+20 HMMT), reasoning (+8 GPQA), instruction following (+13 IFBench).
#   At 5.66 GB it uses a fraction of the VRAM, enabling full 262K native
#   context per slot with 4 parallel slots in only 29.5 GB.
#
# VRAM budget (32,607 MiB total):
#   Model weights:    ~5.66 GB (CUDA)
#   KV cache (1M):    ~17.4 GB (4 slots x 262K, q8_0, 8 attn layers only)
#   Recurrent state:    201 MB (32 DeltaNet layers, fixed size)
#   Compute buffers:  ~6.5 GB (CUDA) + 4.2 GB (Host)
#   Free headroom:    ~2.6 GB
#
# Benchmarked on RTX 5090 (2026-03-02, native sm_120a, BLACKWELL_NATIVE_FP4=1):
#   Single-stream decode:  ~177 tok/s  (compute-bound, 94% SM utilization)
#   4-concurrent aggregate: ~423 tok/s
#   4-concurrent per-slot:  ~106 tok/s each
#   Prompt processing:    ~1,688 tok/s
#   GPU: 100% SM / 63% mem @ 4-concurrent | 94% SM / 65% mem @ single
#   Power: 312W single, 445W 4-concurrent | Temp: 52-60C
# ==============================================================================

PROFILE_CTX_LENGTH=1048576      # 1M total -- 262K per slot with 4 parallel slots
PROFILE_THREADS=16              # Physical cores (avoid hyperthreads for decode)
PROFILE_THREADS_BATCH=20        # Higher thread count for prompt processing
PROFILE_FLASH_ATTN="on"         # Flash attention for long context perf
PROFILE_KV_TYPE_K="q8_0"        # KV cache key quantization
PROFILE_KV_TYPE_V="q8_0"        # KV cache value quantization
PROFILE_NO_MMAP="true"          # Avoid page faults, load model into RAM
PROFILE_JINJA="true"            # Chat template / tool calling support
PROFILE_PARALLEL=4              # 4 slots: 262K/slot, 423 tok/s agg, 106 tok/s each
                                # Dense model: internal --parallel batching is 2.6x more
                                # efficient than running multiple instances (tested)
PROFILE_PRIO=2                  # High thread priority for reduced scheduling latency
PROFILE_CPU_STRICT=1            # Strict CPU placement for cache locality
PROFILE_CACHE_REUSE=0           # Disabled: hybrid recurrent arch re-processes anyway
PROFILE_NO_WEBUI="true"         # Headless: no web UI, reduce attack surface
PROFILE_METRICS="true"          # Prometheus-compatible /metrics endpoint
# --mlock: pin model in RAM; -b/-ub 4096: large batch for fast prompt encode
# --swa-full: full SWA cache for hybrid attention models (DeltaNet + attention)
# --cache-ram 0: disable prompt cache (hybrid recurrent arch forces re-processing)
# --reasoning-format none: keep <think> tags as plain text in content, no reasoning_content field
#   (prevents AI SDK extractReasoningMiddleware crash on empty think blocks)
PROFILE_EXTRA_ARGS="--mlock -b 4096 -ub 4096 --swa-full --cache-ram 0 --reasoning-format none"
