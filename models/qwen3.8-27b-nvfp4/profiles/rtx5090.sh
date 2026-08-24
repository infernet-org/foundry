# shellcheck shell=bash disable=SC2034
# ==============================================================================
# Foundry Profile: RTX 5090 (32GB) -- Qwen3.8-27B-NVFP4 via vLLM
# ==============================================================================
# Unsloth dynamic NVFP4 checkpoint (~22 GB on disk: mixed FP8 channel-wise +
# 4-bit weight groups, BF16 vision tower, built-in MTP draft head).
#
# Architecture: Qwen3_5 dense 27B -- 16/64 full-attention layers + 48
# linear-attention layers (constant recurrent state). Only the full-attention
# layers carry KV cache; FP8 KV doubles capacity vs bf16.
#
# VRAM budget (31.4 GiB usable): weights ~24.6 GiB + KV pool + CUDA graph
# capture all fit BELOW 0.88 utilization. Do NOT raise gpu-memory-utilization
# past ~0.88 on one card: capture allocates outside the budget and OOMs
# (0.93 was verified OOM in the published recipe; 0.88 boots clean).
#
# Benchmarked on RTX 5090 (2026-08-24, vLLM 0.27.1, CUTLASS native-NVFP4
# kernel, MTP x4 speculative decoding, CUDA graphs ON):
#   Single-stream short:      ~157 tok/s
#   Single-stream long-form:  ~122 tok/s sustained (1500-token completion)
#   4-concurrent aggregate:   ~166 tok/s
#   Draft acceptance:          ~0.58 short completions / higher on prose
#   Boot-to-healthy:           ~150 s
#
# Spec-decode trade-off (measured, see EVALUATION.md): MTP multiplies
# SINGLE-stream speed ~2.4x over plain decode (~65 tok/s) but REDUCES
# concurrent aggregate (~210 tok/s plain @ seqs 4) because verification costs
# FLOPs once the GPU is compute-bound -- opposite of the MoE qwen3.6, where
# MTP won both axes. This profile optimizes interactive agents; fleets should
# drop the speculative-config via a custom profile.
#
# Swept and rejected (2026-08-24):
#   eager mode (--enforce-eager)    -- boots where graphs OOM at high util,
#                                      but decode collapses to ~25 tok/s;
#                                      only needed if you raise utilization
#   MTP x2 / x3                     -- 115/134 tok/s single (graphs); x4 wins
#   MTP x4 + async-scheduling       -- 152 tok/s, no gain over plain x4
#   max-num-seqs 4                  -- 91 tok/s @ 4-concurrent; 8 -> 166
#   Inferact/Qwen3.8-27B-NVFP4 A/B  -- identical acceptance (0.58), slower
#                                      (134 tok/s), needs 0.92 util + 8K ctx
#                                      to fit; unsloth mixed build wins
#
# Context ladder on ONE card (262K native; boot-verified 2026-08-24):
#   this profile (graphs+MTP x4)    -- up to 64K via FOUNDRY_CTX_LENGTH
#   eager + MTP (add --enforce-eager)-- 128K boots (145,935-token fp8 pool),
#                                      decode drops to ~50-68 tok/s
#   eager, no MTP, util 0.92        -- 192K boots (248,427-token fp8 pool),
#                                      ~26 tok/s; use default profile +
#                                      FOUNDRY_EXTRA_ARGS overrides
#   >192K                            -- requires TP2 across two cards
#
# Drafter alternatives (boot-tested 2026-08-24 on vLLM 0.27.1):
#   DFlash2 (z-lab/incoai)          -- REJECTED: DFlash2DraftModel not in
#                                      stable vLLM registry (needs SGLang or
#                                      unmerged PR #52816)
#   DSpark (RadixArk)               -- REJECTED: arch resolves but crashes
#                                      ('Qwen3Config' has no hc_mult --
#                                      checkpoint newer than engine support);
#                                      published acceptance was below built-in
#                                      MTP anyway (4.36 vs 5.02 GSM8K len)
# ==============================================================================

PROFILE_CTX_LENGTH=32768        # 32K: fits alongside graphs+MTP; raise w/o MTP
PROFILE_GPU_MEM_UTIL=0.88       # >0.88 risks CUDA-graph-capture OOM on 32GB
PROFILE_MAX_NUM_SEQS=8          # 166 tok/s @4-conc vs 91 at seqs=4
PROFILE_MAX_BATCHED_TOKENS=2048 # Small chunks leave room for graph capture
PROFILE_KV_CACHE_DTYPE=fp8      # ~2x KV capacity vs bf16 on hybrid attention
PROFILE_MOE_BACKEND="auto"      # dense model: vLLM default path
PROFILE_MULTIMODAL="false"      # BF16 vision tower costs VRAM; enable on-demand
# MTP x4 self-speculation (+140% single-stream vs plain graphs)
PROFILE_EXTRA_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":4}"
