# Foundry — agent meta-index

Foundry ships tuned Docker images for NVFP4 inference on Blackwell RTX 50xx /
Hopper GPUs (28 GB+ VRAM). Each model lives in its own directory under
`models/<slug>/` and builds to one image; pass `MODEL=<slug>` to make targets.
OpenAI-compatible API on port 8080. llama.cpp/GGUF support was removed.

| Model slug | Checkpoint | Backend pin |
|------------|-----------|-------------|
| `qwen3.6-35b-a3b-nvfp4` | nvidia/Qwen3.6-35B-A3B-NVFP4 (MoE, ~3B active) | vLLM 0.24.0 |
| `qwen3.8-27b-nvfp4` | unsloth/Qwen3.8-27B-NVFP4 (dense hybrid, MTP head) | vLLM 0.27.1 |

## Repo map

| Path | What it is |
|------|------------|
| `models/<slug>/Dockerfile` | The image. Base is version-pinned per model — do not float to `:latest`. Bumps require a Gate 1 re-sweep of that model |
| `models/<slug>/entrypoint.sh` | GPU detect → profile load → resumable model download → `vllm serve`. 3-tier flags: model defaults → `PROFILE_*` → `FOUNDRY_EXTRA_ARGS` (appends only, cannot remove flags) |
| `models/<slug>/profiles/` | Per-GPU tuning per model. Header comments carry the full sweep record — read them before changing values |
| `EVALUATION.md` | FAP: the 4-gate certification record (throughput / fidelity / quant preservation / measured intelligence) + how to reproduce, per model |
| `scripts/eval/` | FAP gate runners (evalplus, aider polyglot, SWE-bench via mini-SWE-agent) — take the served model id as an argument/default |
| `scripts/download-model.sh` | Host-side weight download (`make download MODEL=<slug>`) |
| `scripts/benchmark.py` | Throughput benchmark (single-stream, prefill, concurrent) |
| `monitoring/` | Prometheus (host port 9091) + Grafana dashboards keyed to `vllm:*` and `nvidia_smi:*` metrics |
| `skills/` | Agent skills — `npx skills add infernet-org/foundry` |
| `AGENTS.md` | How to point agent frameworks AT the served API (integration guide, not repo instructions) |

## Common commands

```bash
make build MODEL=qwen3.8-27b-nvfp4   # build a model image
make run MODEL=qwen3.8-27b-nvfp4     # serve (auto-detects GPU profile); first run downloads ~22 GB
make test MODEL=qwen3.8-27b-nvfp4    # smoke test: boot, health, one completion (allow ~5 min)
make benchmark                       # throughput vs a running server (PORT=8080)
docker compose --profile monitoring up -d   # + Prometheus/Grafana (:3000, admin/admin)
./scripts/eval/run-evalplus.sh <tag>        # FAP Gate 2 fidelity check, ~2 min
```

## Rules that matter (learned the hard way — see EVALUATION.md sweep records)

- **Rerun Gate 2 after ANY serving-config change**: `./scripts/eval/run-evalplus.sh` — 2 minutes, catches silent output corruption from quant/parser/template mistakes.
- **The vLLM base image is pinned per model** because every profile flag was swept against it. Bumping it requires re-running that model's Gate 1 sweep.
- **Qwen3.8 on ONE 32 GB card needs `--enforce-eager`**: CUDA graph capture allocates outside the utilization budget and OOMs next to ~24.6 GiB of NVFP4 weights; only eager decides boot success. TP2 fits graphs again.
- **MTP context trade (qwen3.6)**: the BF16 draft head costs ~1 GB → 224K max ctx with MTP, 262K without. `FOUNDRY_EXTRA_ARGS` cannot disable MTP (append-only); edit the profile.
- **Do not raise `gpu-memory-utilization` past 0.90** on 32 GB cards.
- **Big batches hurt spec decode**: 16 seqs / 8192 batched tokens LOST 32% aggregate in the qwen3.6 sweep. Don't "optimize" that direction without measuring.
- **Quantization is auto-detected on qwen3.8** — do not force `--quantization modelopt` there (Unsloth mixed FP8+NVFP4 repack); set `FOUNDRY_QUANTIZATION` only when debugging.
- Startup takes 2-4 min (weight load + graph capture/eager warmup); health checks and scripts must allow for it.
- Thinking mode: reasoning arrives in `reasoning_content` (never parse `<think>` from content). Disable per request with `"chat_template_kwargs": {"enable_thinking": false}`.
