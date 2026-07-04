---
name: foundry-benchmark
description: Measure foundry's inference throughput (single-stream, prefill, concurrent) and MTP speculative-decoding acceptance against a running server. Use when asked to benchmark, compare configs, or investigate slow inference.
---

# Benchmarking foundry

Requires a healthy server (see foundry-serve).

```bash
python3 scripts/benchmark.py --url http://localhost:8080 --mode all --concurrent 4
```
Modes: `generation` (single-stream), `prompt` (prefill), `throughput` (concurrent), `all`.

## Reference numbers (RTX 5090, rtx5090 profile, MTP x4)
- Single-stream: ~384 tok/s | 4-concurrent: ~1,228 tok/s | VRAM ~29 GB
- First request after boot pays one-time warmup — discard it.

## MTP acceptance (workload-dependent; code > prose)
```bash
curl -s localhost:8080/metrics | grep -E "spec_decode_num_(accepted|draft)_tokens_total"
# acceptance = accepted / draft; sweep baseline was ~0.66
```

If results are >30% below reference: check concurrent load, GPU clocks/temperature,
and that the rtx5090 profile actually loaded (container logs: "Loading profile").
When comparing configs, change ONE flag at a time and rerun; record rejects too —
see the sweep-record format in profiles/rtx5090.sh and EVALUATION.md Gate 1.
