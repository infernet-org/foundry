# ==============================================================================
# Foundry Profile: RTX 5090 (32GB) -- Qwen3.6-35B-A3B-NVFP4 via vLLM
# ==============================================================================
# ModelOpt NVFP4 checkpoint (~22 GB on disk: NVFP4 language model + BF16
# vision tower + FP8 KV-cache scales). Full 262K native context fits: the
# hybrid architecture keeps KV only for the full-attention layers, and FP8
# KV halves it again -- 372K tokens of KV capacity at 0.90 utilization.
#
# Architecture: 40 layers, hybrid Gated DeltaNet (recurrent) + full attention
# with MoE experts (~3B active / 35B total per token)
#
# VRAM budget (32,607 MiB total):
#   Weights + buffers:  ~26,500 MiB (incl. Marlin workspace, CUDA graphs)
#   KV cache (FP8):      ~2,900 MiB (372K tokens capacity)
#   Free headroom:       ~3,200 MiB
#
# Benchmarked on RTX 5090 (2026-07-03, vLLM 0.24.0, MARLIN NvFp4 backend,
# MTP x4 self-speculative decoding + async scheduling):
#   Single-stream decode:   ~384 tok/s  (~210 without MTP)
#   4-concurrent aggregate: ~1,228 tok/s steady (~540 without MTP)
#   Draft acceptance:        ~0.66 (essay text; code accepts higher)
#   Prompt processing:      ~1K tokens in ~0.11s
#   GPU util: 98%  |  Power: ~300W / 575W  |  Temp: 48C  |  VRAM: 29.0 GB
#
# MTP notes: the checkpoint ships an unquantized BF16 MTP head (1 layer,
# reused for all draft tokens). It costs ~1 GB of KV budget, so context
# caps at 224K with MTP (vs 262K without). To trade speed for the full 262K,
# edit PROFILE_EXTRA_ARGS below to drop --speculative-config and raise
# PROFILE_CTX_LENGTH to 262144 (FOUNDRY_EXTRA_ARGS appends flags; it cannot
# remove them). The required --mamba-cache-mode align is added automatically
# by the entrypoint whenever MTP is enabled.
#
# Swept and rejected (2026-07-03):
#   MTP x5                          -- OOM at 224K; acceptance declining
#   16 seqs / 8192 batched tokens   -- concurrent DROPS to 831 tok/s with MTP
#   moe-backend flashinfer_b12x     -- 371/1066 tok/s, no gain over marlin,
#                                      needs draft moe_backend=triton to boot
# ==============================================================================

PROFILE_CTX_LENGTH=229376       # 224K: max that fits with the BF16 MTP head
PROFILE_GPU_MEM_UTIL=0.90       # 0.92+ OOMs during CUDA graph capture on 32GB
PROFILE_MAX_NUM_SEQS=8          # Sweet spot; 16 seqs hurts spec-decode throughput
PROFILE_MAX_BATCHED_TOKENS=4096 # Chunked prefill; also bounds startup profiling memory
PROFILE_MOE_BACKEND="auto"      # auto=MARLIN on sm_120 -- fastest reliable NVFP4 path
PROFILE_MULTIMODAL="false"      # BF16 vision tower costs VRAM; enable on-demand
# MTP x4 self-speculation (+83% single-stream, +127% concurrent) + async sched
PROFILE_EXTRA_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":4} --async-scheduling"
