# Foundry

Tuned Docker images for running open LLMs on consumer GPUs. One command, maximum tok/s.

Foundry provides pre-configured Docker images with per-GPU hardware profiles that automatically detect your GPU and apply optimal inference settings. No manual tuning required.

## Quick Start

```bash
# Run Qwen3.5-35B-A3B with auto-detected GPU settings
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:latest
```

The first run downloads the model (~20GB). Subsequent starts are instant.

Then use it like any OpenAI-compatible API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b-a3b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Works with any OpenAI-compatible client: Cursor, Continue, OpenCode, Open WebUI, etc.

## Supported Hardware

| GPU | VRAM | Expected tok/s | Status |
|-----|------|---------------|--------|
| RTX 5090 | 32 GB | ~160 | Tested |
| RTX 5080 | 16 GB | ~75 | Community |
| RTX 4090 | 24 GB | ~70 | Community |
| RTX 3090 | 24 GB | ~55 | Community |
| A100 80GB | 80 GB | ~80 | Planned |

All numbers are for Qwen3.5-35B-A3B Q4_K_M quantization, single GPU.

## How It Works

Foundry uses [llama.cpp](https://github.com/ggerganov/llama.cpp) as the inference engine. Why not SGLang or vLLM? Because for **consumer GPUs**, llama.cpp's expert-level MoE offloading (`--fit on`) is the only way to run a 35B-parameter MoE model on a single 16-24GB card at full speed. SGLang and vLLM require the entire model to fit in VRAM.

Qwen3.5-35B-A3B is a Mixture-of-Experts model: 35B total parameters but only 3B active per token. llama.cpp keeps attention and norms on GPU while spilling inactive experts to CPU. This is why a 35B MoE model runs **3-10x faster** than a 27B dense model on the same hardware.

### GPU Auto-Detection

On startup, Foundry:
1. Detects your GPU via `nvidia-smi`
2. Loads a tuned hardware profile with optimal settings
3. Downloads the GGUF model if not already cached
4. Launches `llama-server` with the right arguments

### Hardware Profiles

Each profile tunes: context length, KV cache quantization, thread count, memory fraction, flash attention, and MoE offloading strategy.

```bash
# Override auto-detection with a specific profile
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  -e FOUNDRY_PROFILE=rtx4090 \
  ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:latest
```

Available profiles: `rtx5090`, `rtx5080`, `rtx4090`, `rtx3090`, `a100-80g`, `default`

## Build From Source

```bash
# Build the base llama.cpp image
make build-base

# Build the Qwen3.5 model image
make build

# Run locally
make run
```

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FOUNDRY_PROFILE` | `auto` | GPU profile (`auto`, `rtx5090`, `rtx4090`, etc.) |
| `FOUNDRY_PORT` | `8080` | Server port |
| `FOUNDRY_CTX_LENGTH` | Profile default | Context window size |
| `FOUNDRY_THREADS` | Profile default | CPU threads for expert offloading |
| `FOUNDRY_EXTRA_ARGS` | `` | Additional llama-server arguments |

## Architecture

```
foundry/
├── base/llama-cpp/          # Base image: llama.cpp + CUDA runtime
├── models/
│   └── qwen3.5-35b-a3b/    # Model-specific image + profiles
│       ├── Dockerfile
│       ├── entrypoint.sh
│       └── profiles/        # Per-GPU tuned launch configs
├── scripts/                 # Build, run, download helpers
└── docker-compose.yml       # Easy local deployment
```

## Models

### Qwen3.5-35B-A3B

- **Architecture**: Hybrid Gated DeltaNet + MoE (35B total, 3B active)
- **Quantization**: Q4_K_M via [unsloth](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)
- **Disk size**: ~20 GB
- **Min VRAM**: 16 GB (with expert offloading)
- **Context**: Up to 262K native, default varies by GPU profile

## License

Apache-2.0
