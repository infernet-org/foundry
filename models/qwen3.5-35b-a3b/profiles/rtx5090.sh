# ==============================================================================
# Foundry Profile: RTX 5090 (32GB)
# ==============================================================================
# Qwen3.5-35B-A3B UD-Q4_K_XL (~20.6GB) fits entirely in VRAM.
# All layers on GPU. No expert offloading needed.
#
# Architecture: 30 Gated DeltaNet (recurrent) + 10 full attention layers
# with 256 MoE experts (top-8 + 1 shared active per token, ~3B active)
#
# VRAM budget (32,607 MiB total):
#   Model weights:    19,219 MiB (CUDA) + 398 MiB (Host)
#   KV cache (192K):  ~2,040 MiB (10 attn layers only, q8_0)
#   Recurrent state:     ~93 MiB (30 DeltaNet layers, fixed size)
#   Compute buffers:   3,944 MiB (CUDA) + 2,112 MiB (Host)
#   Free headroom:    ~5,300 MiB
#
# Benchmarked on RTX 5090 (2026-02-27):
#   Single-stream decode:  ~174 tok/s  (memory-bandwidth-bound @ 49% util)
#   4-concurrent aggregate: ~320 tok/s (+84% via MoE expert batching)
#   Prompt processing:    ~1,163 tok/s (internal metric)
#   GPU util: 92%  |  Power: 337W / 575W  |  Temp: 52C
#
# Tested and rejected (no measurable impact on this memory-bound workload):
#   --poll 100, --prio-batch 2, --flash-attn auto, nvidia-smi -lgc 3105
# ==============================================================================

PROFILE_CTX_LENGTH=196608       # 192K context -- uses ~2GB KV, fits with 5GB headroom
PROFILE_THREADS=16              # Physical cores (avoid hyperthreads for decode)
PROFILE_THREADS_BATCH=20        # Higher thread count for prompt processing
PROFILE_FLASH_ATTN="on"         # Flash attention for long context perf
PROFILE_KV_TYPE_K="q8_0"        # KV cache key quantization
PROFILE_KV_TYPE_V="q8_0"        # KV cache value quantization
PROFILE_NO_MMAP="true"          # Avoid page faults, load model into RAM
PROFILE_JINJA="true"            # Chat template / tool calling support
PROFILE_PARALLEL=4              # 4 concurrent slots: +84% aggregate throughput via MoE batching
                                # (CUDA graphs for MUL_MAT_ID at BS 1-4, PR #19645)
                                # Single-stream decode unchanged at ~174 tok/s, only +840 MiB VRAM
PROFILE_PRIO=2                  # High thread priority for reduced scheduling latency
PROFILE_CPU_STRICT=1            # Strict CPU placement for cache locality
PROFILE_CACHE_REUSE=256         # KV cache reuse for multi-turn chat prefix sharing
PROFILE_NO_WEBUI="true"         # Headless: no web UI, reduce attack surface
PROFILE_METRICS="true"          # Prometheus-compatible /metrics endpoint
# --mlock: pin model in RAM; -b/-ub 4096: large batch for fast prompt encode
# --swa-full: full SWA cache for hybrid attention models
# --cache-ram 0: disable prompt cache (Qwen3.5 hybrid recurrent arch forces re-processing anyway)
PROFILE_EXTRA_ARGS="--mlock -b 4096 -ub 4096 --swa-full --cache-ram 0"
