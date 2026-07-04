# Foundry — agent meta-index

Foundry ships ONE tuned Docker image: **nvidia/Qwen3.6-35B-A3B-NVFP4** served
by **vLLM 0.24.0** (pinned) on NVFP4-capable GPUs (Blackwell RTX 50xx / Hopper,
32 GB+ VRAM). OpenAI-compatible API on port 8080. There are no other models in
this repo; llama.cpp/GGUF support was removed.

## Repo map

| Path | What it is |
|------|------------|
| `models/qwen3.6-35b-a3b-nvfp4/Dockerfile` | The image. Base is version-pinned — do not float to `:latest` |
| `models/qwen3.6-35b-a3b-nvfp4/entrypoint.sh` | GPU detect → profile load → resumable model download → `vllm serve`. 3-tier flags: model defaults → `PROFILE_*` → `FOUNDRY_EXTRA_ARGS` (appends only, cannot remove flags) |
| `models/qwen3.6-35b-a3b-nvfp4/profiles/` | Per-GPU tuning. `rtx5090.sh` ships MTP x4 speculative decoding: 384 tok/s single / 1,228 tok/s 4-concurrent, 224K ctx. Header comments carry the full sweep record — read them before changing values |
| `EVALUATION.md` | FAP: the 4-gate certification record (throughput / fidelity / quant preservation / measured intelligence) + how to reproduce |
| `scripts/eval/` | FAP gate runners (evalplus, aider polyglot, SWE-bench via mini-SWE-agent) |
| `scripts/benchmark.py` | Throughput benchmark (single-stream, prefill, concurrent) |
| `monitoring/` | Prometheus (host port 9091) + Grafana dashboards keyed to `vllm:*` and `nvidia_smi_*` metrics |
| `skills/` | Agent skills — `npx skills add infernet-org/foundry` |
| `AGENTS.md` | How to point agent frameworks AT the served API (integration guide, not repo instructions) |

## Common commands

```bash
make build          # build the image
make run            # serve (auto-detects GPU profile); first run downloads ~22 GB
make test           # smoke test: boot, health, one completion (allow ~5 min)
make benchmark      # throughput vs a running server (PORT=8080)
docker compose --profile monitoring up -d   # + Prometheus/Grafana (:3000, admin/admin)
./scripts/eval/run-evalplus.sh <tag>        # FAP Gate 2 fidelity check, ~2 min
```

## Rules that matter (learned the hard way — see EVALUATION.md sweep record)

- **Rerun Gate 2 after ANY serving-config change**: `./scripts/eval/run-evalplus.sh` — 2 minutes, catches silent output corruption from quant/parser/template mistakes.
- **The vLLM base image is pinned** because every profile flag was swept against it. Bumping it requires re-running the Gate 1 sweep.
- **MTP context trade**: the BF16 draft head costs ~1 GB → 224K max ctx with MTP, 262K without. `FOUNDRY_EXTRA_ARGS` cannot disable MTP (append-only); edit the profile.
- **Do not raise `gpu-memory-utilization` past 0.90** on 32 GB cards — CUDA graph capture OOMs.
- **Big batches hurt spec decode**: 16 seqs / 8192 batched tokens LOST 32% aggregate in the sweep. Don't "optimize" that direction without measuring.
- Startup takes 2-4 min (weight load + CUDA graph capture); health checks and scripts must allow for it.
- Thinking mode: reasoning arrives in `reasoning_content` (never parse `<think>` from content). Disable per request with `"chat_template_kwargs": {"enable_thinking": false}`.
