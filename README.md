# Foundry

Tuned Docker images for running open LLMs on consumer GPUs. One command, maximum tok/s.

Foundry provides pre-configured Docker images with per-GPU hardware profiles that automatically detect your GPU and apply optimal inference settings. No manual tuning required.

## Quick Start

```bash
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:latest
```

Or with Docker Compose:

```bash
docker compose up
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

| GPU | VRAM | Context | Decode | 4-concurrent |
|-----|------|---------|--------|--------------|
| RTX 5090 | 32 GB | 192K | ~174 tok/s | ~320 tok/s |
| Other NVIDIA (16GB+) | 16+ GB | 16K | varies | varies |

*Benchmarked with `Qwen3.5-35B-A3B` using `UD-Q4_K_XL` quantization (Unsloth Dynamic 2.0).*

### Hermes-4.3-36B (Dense)
| GPU | VRAM | Context | Decode | 4-concurrent |
|-----|------|---------|--------|--------------|
| RTX 5090 | 32 GB | 32K | ~64.5 tok/s | ~132.0 tok/s |
| Other NVIDIA (24GB+) | 24+ GB | 8K | varies | varies |

*Benchmarked with `NousResearch/Hermes-4.3-36B` using `Q4_K_M` quantization.*

### Hermes-4.3-36B (Dense)
| GPU | VRAM | Context | Decode | 2-concurrent |
|-----|------|---------|--------|--------------|
| RTX 5090 | 32 GB | 32K | ~29 tok/s | ~54 tok/s |
| Other NVIDIA (24GB+) | 24+ GB | 8K | varies | varies |

*Benchmarked with `NousResearch/Hermes-4.3-36B` using `Q4_K_M` quantization.*

## How It Works

Foundry uses [llama.cpp](https://github.com/ggml-org/llama.cpp) as the inference engine, built on the official [`server-cuda12`](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp) image.

Why not SGLang or vLLM? For **consumer GPUs**, llama.cpp's MoE expert offloading (`--fit on`) is the only engine that can run a 35B-parameter MoE model on a single 16-24GB card at full speed. SGLang and vLLM require the entire model to fit in VRAM.

Qwen3.5-35B-A3B is a Mixture-of-Experts model: 35B total parameters but only 3B active per token. llama.cpp keeps attention layers on GPU while spilling inactive experts to CPU, which is why a 35B MoE runs **faster** than a 27B dense model on the same hardware.

### GPU Auto-Detection

On startup, Foundry:
1. Detects your GPU via `nvidia-smi`
2. Loads a tuned hardware profile with optimal settings
3. Downloads the GGUF model if not already cached
4. Launches `llama-server` with the right arguments

### Hardware Profiles

Each profile tunes: context length, KV cache quantization, thread count, batch size, flash attention, thread priority, CPU affinity, and Prometheus metrics.

```bash
# Override auto-detection with a specific profile
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  -e FOUNDRY_PROFILE=rtx5090 \
  ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:latest
```

Available profiles: `rtx5090`, `default`

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FOUNDRY_PROFILE` | `auto` | GPU profile (`auto`, `rtx5090`, `default`) |
| `FOUNDRY_PORT` | `8080` | Server port |
| `FOUNDRY_CTX_LENGTH` | Profile default | Context window size |
| `FOUNDRY_THREADS` | Profile default | CPU thread count |
| `FOUNDRY_EXTRA_ARGS` | (empty) | Additional llama-server arguments (highest priority) |
| `HF_TOKEN` | (empty) | Hugging Face token for authenticated downloads |

## Multi-Agent Inference

The RTX 5090 profile is configured with `--parallel 4`, enabling 4 concurrent inference slots. This makes Foundry well-suited for multi-agent workflows where several AI agents share a single GPU and model.

### Why this works

Qwen3.5-35B-A3B uses a 256-expert Mixture-of-Experts architecture with only 8 experts active per token. During single-stream decode, the GPU's tensor cores are largely idle -- the bottleneck is memory bandwidth, not compute. When multiple agents send concurrent requests, llama.cpp batches token generation across all active slots. Different tokens may route to different experts, and CUDA graphs (for `MUL_MAT_ID` at batch size 1-4) capture the entire batched MoE operation, significantly improving GPU utilization.

### Empirically validated throughput

| Active agents | Aggregate throughput | Per-agent speed | VRAM |
|---------------|---------------------|-----------------|------|
| 1 | 174 tok/s | 174 tok/s | 25.3 GB |
| 2 | 234 tok/s | ~117 tok/s each | 25.7 GB |
| 4 | 320 tok/s | ~80 tok/s each | 26.1 GB |

Single-agent speed is unaffected. The 4 slots only activate when there are concurrent requests.

### Compatible frameworks

Any OpenAI-compatible agent framework works out of the box -- point it at `http://localhost:8080/v1`:

- [OpenCode](https://opencode.ai) / [Cursor](https://cursor.com) / [Continue](https://continue.dev) -- coding agents
- [CrewAI](https://crewai.com) / [AutoGen](https://github.com/microsoft/autogen) -- multi-agent orchestration
- [Open WebUI](https://openwebui.com) -- chat interface with multi-user support

### Scaling to 2 GPUs

With 2x RTX 5090, run two independent Foundry instances (one per GPU) for 8 total concurrent slots and 348 tok/s combined throughput with zero contention between agents:

```bash
# GPU 0: agents 1-4
docker run --gpus '"device=0"' -p 8080:8080 -v ~/.cache/foundry:/models \
  ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:latest

# GPU 1: agents 5-8
docker run --gpus '"device=1"' -p 8081:8080 -v ~/.cache/foundry:/models \
  ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:latest
```

## Docker Compose

```bash
# Basic
docker compose up

# With explicit profile
FOUNDRY_PROFILE=rtx5090 docker compose up

# With monitoring stack (Prometheus + Grafana + GPU metrics)
docker compose --profile monitoring up
```

Create a `.env` file for secrets and optional monitoring config:

```
HF_TOKEN=hf_your_token_here
GF_ADMIN_USER=admin
GF_ADMIN_PASSWORD=admin
```

## Monitoring

Foundry includes an optional observability stack activated via Docker Compose profiles. It scrapes inference metrics from llama-server, GPU telemetry via nvidia-smi, host resources, and container stats -- all visualized in pre-configured Grafana dashboards.

```bash
docker compose --profile monitoring up
```

Then open:
- **Grafana**: [http://localhost:3000](http://localhost:3000) (default: admin / admin)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)

### What Gets Monitored

| Layer | Source | Metrics |
|-------|--------|---------|
| Inference | llama-server `/metrics` | Decode tok/s, prompt tok/s, active slots, deferred requests, total tokens, decode calls |
| GPU | nvidia-gpu-exporter | VRAM usage, GPU utilization, memory bandwidth, temperature, power, clock speeds, fan |
| Host | node-exporter | CPU, RAM, disk, network, load average |
| Container | cAdvisor | Per-container CPU, memory, network I/O |

### Pre-Configured Dashboards

| Dashboard | Description |
|-----------|-------------|
| **Foundry Inference** | Custom dashboard: inference throughput gauges, slot utilization, GPU telemetry, host resources |
| **Node Exporter Full** | Comprehensive host metrics (community dashboard #1860) |
| **NVIDIA GPU** | Detailed GPU monitoring (community dashboard #14574) |
| **cAdvisor** | Docker container resources (community dashboard #14282) |

All dashboards are auto-provisioned on first start -- no manual import needed.

### Architecture

```
┌─────────────────┐     ┌────────────────┐     ┌─────────┐
│  llama-server    │────▶│   Prometheus    │────▶│ Grafana │
│  :8080/metrics   │     │   :9090         │     │ :3000   │
├─────────────────┤     │                │     └─────────┘
│  nvidia-gpu-exp  │────▶│  scrapes every  │
│  :9835           │     │  15 seconds     │
├─────────────────┤     │                │
│  node-exporter   │────▶│  30-day         │
│  :9100           │     │  retention      │
├─────────────────┤     │                │
│  cAdvisor        │────▶│                │
│  :8081           │     └────────────────┘
└─────────────────┘
```

## Host Kernel Tuning (Optional)

For maximum performance, run the host tuning script once on the Docker host:

```bash
sudo ./scripts/host-setup.sh
```

This tunes: `vm.swappiness`, `vm.overcommit_memory`, hugepages, TCP buffers, CPU governor, and NVIDIA persistence mode. Changes are not persistent across reboots -- the script prints instructions for making them permanent.

## Build From Source

```bash
make build    # Build the model image
make run      # Run with auto-detected GPU
make test     # Smoke test: start, wait for health, send one request
make download # Download the GGUF model file to ~/.cache/foundry
```

## Architecture

```
foundry/
├── models/
│   └── qwen3.5-35b-a3b/
│       ├── Dockerfile           # FROM llama.cpp:server-cuda12
│       ├── entrypoint.sh        # GPU detect, model download, launch
│       └── profiles/
│           ├── rtx5090.sh       # 192K ctx, 4 slots, 320 tok/s aggregate
│           └── default.sh       # 16K ctx, q4_0 KV, conservative
├── scripts/
│   ├── benchmark.py             # Generation speed, prompt processing, throughput
│   ├── optimize_5090.py         # Multi-config A/B testing harness
│   ├── download-model.sh        # Download GGUF outside Docker
│   └── host-setup.sh            # Linux kernel tuning for inference
├── docker-compose.yml
├── Makefile
└── .github/workflows/build.yml  # CI: build and push to GHCR
```

## Benchmark

RTX 5090 profile results (Qwen3.5-35B-A3B UD-Q4_K_XL, 192K context, 4 slots):

```
SINGLE-STREAM DECODE:    ~174 tok/s
4-CONCURRENT AGGREGATE:  ~320 tok/s  (+84% via MoE expert batching)
PROMPT PROCESSING:     ~1,163 tok/s  (internal metric)
GPU UTILIZATION:            92%
MEMORY BANDWIDTH:           49%      (bottleneck: 878 / 1,792 GB/s)
POWER DRAW:                337W / 575W TDP
TEMPERATURE:                52C      (under sustained load)
VRAM USAGE:              26.1 GB / 32.6 GB (6 GB headroom)
```

Run your own benchmark:

```bash
python3 scripts/benchmark.py --url http://localhost:8080 --mode all
```

## Models

### Qwen3.5-35B-A3B

- **Architecture**: Hybrid Gated DeltaNet + MoE (35B total, 3B active per token)
  - 30 recurrent layers (Gated DeltaNet, fixed-size state, no KV cache)
  - 10 full attention layers (standard KV cache, GQA 8:1)
  - 256 experts per MoE layer, top-8 + 1 shared active per token
- **Quantization**: UD-Q4_K_XL via [Unsloth](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) (Dynamic 2.0)
- **Disk size**: ~20.6 GB
- **Min VRAM**: 16 GB (with expert offloading)
- **Max context**: 262K native, 192K default on RTX 5090

### Hermes-4.3-36B

- **Architecture**: Dense (36B total, all 36B active per token)
  - ByteDance Seed-OSS-36B architecture
  - Standard attention (GQA 80:8)
- **Quantization**: Q4_K_M via [bartowski](https://huggingface.co/bartowski/NousResearch_Hermes-4.3-36B-GGUF)
- **Disk size**: ~21.8 GB
- **Min VRAM**: 24 GB (dense models cannot effectively offload experts)
- **Max context**: 512K native, 32K default on RTX 5090

## License

Apache-2.0
