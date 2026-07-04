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
rtx5090 profile** (vLLM 0.24.0, RTX 5090 32 GB, 2026-07-03).

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

*(Not yet run for this deployment -- harness is prepped; run the seeded slice below to fill this gate.)*

```bash
./scripts/eval/run-swebench.sh 0:50     # seeded slice (~2-3 h)
./scripts/eval/run-swebench.sh 0:500    # full Verified (overnight)
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
