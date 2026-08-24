# FAP -- The Foundry Assessment Protocol

No configuration ships without passing four gates, in order:

```
 GATE 1  THROUGHPUT CEILING     sweep the config space, find the pareto edge
 GATE 2  DEPLOYMENT FIDELITY    prove the serving stack does not corrupt output
 GATE 3  QUANT PRESERVATION     prove quantization did not degrade the model
 GATE 4  MEASURED INTELLIGENCE  rank real SWE capability + token efficiency
```

Each gate has a runner in `scripts/eval/`, a pass criterion, and a results
artifact. Below: the certification record for **qwen3.6-35b-a3b-nvfp4 /
rtx5090 profile** (vLLM 0.24.0, RTX 5090 32 GB, 2026-07-03), followed by the
**qwen3.8-27b-nvfp4 / rtx5090 profile** record (vLLM 0.27.1, RTX 5090 32 GB,
2026-08-24).

---

# Certification: qwen3.8-27b-nvfp4 (2026-08-24)

Checkpoint: [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
(mixed FP8 channel-wise + NVFP4 weight groups, built-in MTP head). Base image
`vllm/vllm-openai:v0.27.1`, `transformers>=5.8.0`. Quantization auto-detected.

## Gate 1 -- Throughput ceiling

Full sweep on RTX 5090 (32 GB), single card. Raw artifacts:
`/tmp` sweep logs reproduced in the `rtx5090.sh` profile header; acceptance from
`vllm:spec_decode_num_{accepted,draft}_tokens_total`.

| Config | Single-stream | 4-concurrent | Accept rate | Verdict |
|--------|---------------|--------------|-------------|---------|
| eager, no MTP (32K) | 25.5 tok/s | 91 tok/s | -- | rejected (launch-bound) |
| eager + MTP x2 | 50.6 | 123 | ~0.7 | rejected |
| eager + MTP x3 | 63.9 | 141 | 0.68 | rejected |
| eager + MTP x4 | 68.1 | 79 | 0.58 | rejected (conc. collapse) |
| graphs, no MTP (32K) | 65.6 | 210-214 | -- | reference |
| graphs + MTP x2 | 114.8 | 105.8 | -- | rejected (conc.) |
| graphs + MTP x3 | 133.8 | 108.2 | -- | rejected (conc.) |
| **graphs + MTP x4, seqs 8** | **157.0** | **166.1** | **0.58** | **SHIPPED** |
| graphs + MTP x4 + async | 152.0 | 90.4 | -- | rejected (no gain) |
| Inferact W4A4 build A/B (x4) | 134.4 @8K | 74.1 | 0.58 | rejected (needs util 0.92 + 8K ctx to boot; identical drafter quality) |

Key finding: unlike MoE qwen3.6 (MTP wins both axes), this dense hybrid is
compute-bound at concurrency -- speculation trades aggregate for latency.
Shipped profile optimizes interactive agents; long-form sustained decode
measured **122 tok/s** (1500-token completion). Context ladder on one card:
64K with the shipped profile, 128K by adding `--enforce-eager`
(145,935-token fp8 pool verified), 192K eager + no-MTP + util 0.92
(248,427-token pool, ~26 tok/s); beyond that requires TP2 (recipe-verified 262K).

Drafter alternatives boot-tested on stable vLLM 0.27.1 and rejected:
`DFlash2DraftModel` is absent from the engine registry (SGLang or unmerged
vLLM PR #52816 required), and RadixArk DSpark resolves but crashes on a
checkpoint/engine config mismatch (`hc_mult`) with published acceptance below
the built-in MTP anyway (4.36 vs 5.02 GSM8K acceptance-length).

## Gate 2 -- Deployment fidelity

```bash
FOUNDRY_EXTRA_ARGS='--default-chat-template-kwargs {"enable_thinking":false}' <server>
./scripts/eval/run-evalplus.sh qwen38-gate2 http://localhost:8090/v1 qwen3.8-27b-nvfp4
```

Preflight thinking-off probe passed. HumanEval+ greedy pass@1:

| Suite | qwen3.8-27b-nvfp4 | qwen3.6-35b-a3b-nvfp4 (ref) |
|-------|-------------------|------------------------------|
| HumanEval (base) | 93.3% | 92.7% |
| HumanEval+ | **90.9%** | 88.4% |

**PASS** -- the vLLM 0.27.1 / auto-quant / qwen3_coder-parser chain does not
corrupt output.

## Gate 3 -- Quantization preservation

BF16 needs ~56 GB VRAM (no local run); comparison is the quantizer's published
logit-level analysis ([unsloth accuracy tables](https://unsloth.ai/docs/models/qwen3.8)),
cross-checked against our Gate 2 absolute score.

| Evidence | Value |
|----------|-------|
| Top-1 agreement vs BF16 (code/chat/multiling.) | 92.2-96.7% |
| KLD mean vs BF16 | 0.012-0.058 across domains |
| Published speedup vs BF16 | 1.49x single / 1.41-1.45x batched (B200) |
| Our HumanEval+ vs published BF16-class agentic scores | consistent (90.9% HE+) |

**PASS with caveat** -- unsloth's mixed FP8+NVFP4 recovers 92-97% top-1
agreement (their dynamic-quant methodology trades a few points of logit
fidelity for ~1.5x speed and ~2x KV capacity vs uniform builds). Benchmark-
level behavior is preserved per Gate 2; logit-level recovery is below the
99% benchmark-preservation bar qwen3.6 set only because the metric differs.
Size reduction: 56 GB BF16 -> 22 GB.

---


## Gate 1 -- Throughput ceiling

Staged sweep with `scripts/benchmark.py`: warmup, 3x 512-token single-stream,
4-concurrent steady-state, draft acceptance from vLLM `/metrics`. Rejected
configs are part of the record -- a config is only "best" relative to what it beat.

| Config | Single-stream | 4-concurrent steady | Accept rate | Verdict |
|--------|---------------|---------------------|-------------|---------|
| baseline (no MTP) | 210 tok/s | 540 tok/s | -- | reference |
| MTP x1 | 296 | 691 | 0.90 | |
| MTP x2 | 319 | 683 | 0.79 | |
| MTP x3 | 364 | 1,120 | 0.70 | |
| MTP x3 + async | 369 | 1,152 | 0.70 | |
| **MTP x4 + async** | **384** | **1,228** | 0.66 | **shipped** |
| MTP x5 | OOM @224K | -- | -- | rejected |
| MTP x4 + 16 seqs / 8K batch | 374 | 831 | 0.61 | rejected: batching fights spec decode |
| MTP x4 + b12x target / triton draft | 371 | 1,066 | 0.62 | rejected: no gain over marlin |

Findings:

- **MTP self-speculation dominates**: the checkpoint ships its own BF16 draft
  head. 1.9x single-stream, 2.3x concurrent. Cost: ~1 GB KV -> context caps
  at 224K (vs 262K plain).
- Acceptance decays with draft depth (0.90 -> 0.66) but net throughput rises
  through x4; x5 no longer fits in 32 GB.
- Bigger batching backfires under spec decode: 16 seqs / 8192 batched tokens
  *lost* 32% aggregate.
- MARLIN is the right NVFP4 MoE kernel on consumer Blackwell (sm_120).

## Gate 2 -- Deployment fidelity

HumanEval+ **greedy** against the live endpoint. Greedy + speculative decoding
is mathematically lossless, so an in-band score certifies the whole chain
(quant kernels, FP8 KV, MTP, reasoning parser, chat template) at once.
Costs 2 minutes -- rerun after any config change.

| Benchmark | pass@1 | Expected band | Verdict |
|-----------|--------|---------------|---------|
| HumanEval | **91.5%** | ~90% | PASS |
| HumanEval+ (extra tests) | **88.4%** | ~85-89% | PASS |

```bash
# Gate 2 requires a thinking-off serving config (evalplus caps generation at
# 768 tokens; thinking would consume the budget and return null content):
FOUNDRY_EXTRA_ARGS='--default-chat-template-kwargs {"enable_thinking":false}' make run
./scripts/eval/run-evalplus.sh gate2
```

**Best-of-N rider** (validated 2026-07-04): 6 samples @ temp 0.8 + execution
selection = **90.9%** vs 88.4% greedy (oracle pass@6: 93.3%). 984 samples in
69 s -- the concurrent MTP throughput makes N=6 near-free in wall-clock. Gains
scale with task headroom; on agentic SWE tasks expect the 10-20 pt regime.

## Gate 3 -- Quantization preservation

The BF16 original needs ~70 GB VRAM; the comparison is the quantizer's
published table ([model card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)),
cross-checked against our Gate-2 result. Pass: >=99% preservation.

| Benchmark | BF16 | NVFP4 | Preservation |
|-----------|------|-------|--------------|
| MMLU Pro | 85.6 | 85.0 | 99.3% |
| GPQA Diamond | 84.9 | 84.8 | 99.9% |
| AIME 2025 | 89.2 | 88.8 | 99.6% |
| SciCode | 40.8 | 40.6 | 99.5% |
| τ²-Bench Telecom | 95.5 | 94.7 | 99.2% |
| IFBench | 62.3 | 62.8 | 100.8% |
| MMMU PRO | 74.1 | 74.5 | 100.5% |
| AA-LCR | 62.0 | 62.0 | 100% |

**PASS** -- 3.06x size reduction (70 GB -> 22 GB) costs ~0-1%.

## Gate 4 -- Measured intelligence

### 4a. Aider polyglot (225 Exercism tasks, 6 languages)

Coding-assistant behavior with per-task token accounting -- FAP's
token-efficiency instrument. Run thinking off vs on for the
pass-rate-per-token frontier: thinking bought **+11 pts pass@2 for 4.3x the tokens**.

| Configuration | pass@2 | pass@1 | Well-formed | Completion tokens | Wall clock |
|---------------|--------|--------|-------------|-------------------|------------|
| Thinking OFF | **50.2%** | 25.8% | 94.7% | 697K (~3.1K/case) | ~29 min |
| Thinking ON (partial: 206/225, run stopped) | **61.2%** | 34.0% | 99.0% | 2.75M (~13.4K/case) | ~40 s/case |

```bash
./scripts/eval/run-aider.sh gate4a
```

### 4b. SWE-bench Verified (mini-SWE-agent scaffold)

Real GitHub issues resolved in real repos -- directly comparable to published
DeepSWE / Qwen / GLM numbers. The minimal bash-only scaffold measures the
*model*, not a product harness. Runner prices output at `1e-6`/token so
`instance_cost * 1e6` = exact output tokens per task -> **tokens per solved
issue**.

Result (RTX 5090, MTP x4 + tool calling, thinking on, seeded 49/50 slice --
one instance dropped to an infra hang; 2026-07-04):

| Cut | Value |
|-----|-------|
| **Resolved (headline)** | **24/49 = 49.0%** |
| Of patches that applied | 24/36 = 66.7% (fix-quality) |
| Malformed submissions | 10/49 (wrote the fix, botched the git-diff submit protocol) |
| Gave up | 3 |
| Output tokens / solved issue | ~12,100 |

Read: fix-quality is mid-level (2 of 3 applied patches pass, incl. correct
root-cause fixes in Django ORM internals); the weakness is agentic-protocol
reliability -- 20% of runs failed on submission format, not correctness, so
that is the highest-leverage thing to fix. Caveats: n=49 (wide CI ~+/-14%),
Django-heavy draw (15 of 24 resolved), and contamination-suspect on a
mid-2026 model vs 2023-era issues -- SWE-rebench is the clean cross-check.
Even discounted, ~49% is at/above DeepSWE-Preview (42.2%, a 32B that needed
RL) from an untrained off-the-shelf checkpoint on one consumer GPU.

```bash
# 1. rollout (thinking on for agentic quality)
FOUNDRY_EXTRA_ARGS='' ./scripts/eval/run-swebench.sh 0:50    # seeded slice
./scripts/eval/run-swebench.sh 0:500                          # full Verified (overnight)

# 2. grade (rootless docker needs DOCKER_HOST set; native docker does not)
DOCKER_HOST="unix://$XDG_RUNTIME_DIR/docker.sock" \
  "$FAP_EVAL_HOME/venv/bin/python" -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path results/swebench-0-50/preds.json \
  --max_workers 4 --run_id foundry-gate4b --cache_level env
```

---

All gates run CPU-side against the serving endpoint -- the GPU stays dedicated
to inference.

One-time harness setup (override the location with `FAP_EVAL_HOME`):

```bash
EVAL=${FAP_EVAL_HOME:-$HOME/.cache/foundry/eval}
python3 -m venv "$EVAL/venv" && "$EVAL/venv/bin/pip" install evalplus mini-swe-agent
git clone https://github.com/Aider-AI/aider "$EVAL/aider"
git clone https://github.com/Aider-AI/polyglot-benchmark "$EVAL/aider/tmp.benchmarks/polyglot-benchmark"
(cd "$EVAL/aider" && ./benchmark/docker_build.sh)   # aider-benchmark image (language toolchains)
cat > "$EVAL/litellm-registry.json" <<'EOF'
{"hosted_vllm/qwen3.6-35b-a3b-nvfp4": {"max_tokens": 32768, "max_input_tokens": 196608,
 "max_output_tokens": 32768, "input_cost_per_token": 0.0, "output_cost_per_token": 0.000001,
 "litellm_provider": "hosted_vllm", "mode": "chat"}}
EOF
```
