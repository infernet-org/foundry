# Foundry

Tuned Docker images for running open LLMs on consumer GPUs. One command, maximum tok/s.

Foundry compiles [llama.cpp](https://github.com/ggml-org/llama.cpp) from source for native Blackwell (sm_120a) and Ada (sm_89) GPU architectures, bundles per-GPU hardware profiles, and auto-detects your GPU at startup. No manual tuning required.

## Table of Contents

- [Quick Start](#quick-start)
- [Models](#models)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Multi-Agent Inference](#multi-agent-inference)
- [Running](#running)
- [Monitoring](#monitoring)
- [Host Kernel Tuning](#host-kernel-tuning)
- [Project Structure](#project-structure)
- [License](#license)

## Quick Start

```bash
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:latest
```

The first run downloads the model (~20 GB). Subsequent starts are instant.

Then use it like any OpenAI-compatible API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b-a3b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Works with any OpenAI-compatible client: Cursor, Continue, OpenCode, Open WebUI, CrewAI, AutoGen, etc. See [AGENTS.md](AGENTS.md) for detailed integration guides.

## Models

### Qwen3.5-35B-A3B (MoE)

Hybrid Gated DeltaNet + Mixture-of-Experts. 35B total parameters, only 3B active per token.

- 30 recurrent layers (Gated DeltaNet -- fixed-size state, no KV cache)
- 10 full attention layers (standard KV cache, GQA 8:1)
- 256 experts per MoE layer, top-8 + 1 shared active per token
- Quantization: UD-Q4_K_XL via [Unsloth](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) (Dynamic 2.0)
- Disk: ~20.6 GB | Min VRAM: 16 GB (with expert offloading) | Max context: 262K native

| GPU | VRAM | Context | Decode | 4-concurrent | VRAM used |
|-----|------|---------|--------|--------------|-----------|
| RTX 5090 | 32 GB | 192K | ~181 tok/s | ~320 tok/s | 26.1 GB |
| Other NVIDIA (16 GB+) | 16+ GB | 16K | varies | varies | varies |

<details>
<summary>RTX 5090 detailed benchmark</summary>

```
SINGLE-STREAM DECODE:    ~181 tok/s
4-CONCURRENT AGGREGATE:  ~320 tok/s  (+77% via MoE expert batching)
PROMPT PROCESSING:     ~1,163 tok/s  (internal metric)
GPU UTILIZATION:            92%
MEMORY BANDWIDTH:           49%      (bottleneck: 878 / 1,792 GB/s)
POWER DRAW:                312W / 575W TDP
TEMPERATURE:                52C      (under sustained load)
VRAM USAGE:              26.1 GB / 32.6 GB (6 GB headroom)
```

Benchmarked 2026-03-01 with native sm_120a (Blackwell) compilation and `BLACKWELL_NATIVE_FP4=1` enabled.
</details>

### Hermes-4.3-36B (Dense)

Dense transformer. 36B total parameters, all active per token. ByteDance Seed-OSS-36B architecture.

- 64 transformer layers, standard attention (GQA 80:8)
- Quantization: Q4_K_M via [NousResearch](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF)
- Disk: ~21.8 GB | Min VRAM: 24 GB | Max context: 512K native

| GPU | VRAM | Context | Decode | 4-concurrent | VRAM used |
|-----|------|---------|--------|--------------|-----------|
| RTX 5090 | 32 GB | 32K | ~64 tok/s | ~132 tok/s | 27.8 GB |
| Other NVIDIA (24 GB+) | 24+ GB | 8K | varies | varies | varies |

Dense models activate all parameters per token, making them compute-bound rather than memory-bandwidth-bound. Expect ~3x slower decode than equivalently-sized MoE models on the same hardware.

### Qwen3-Coder-30B-A3B (MoE)

Standard Mixture-of-Experts optimized for code generation. 30B total parameters, only 3B active per token.

- 48 transformer layers, standard attention (GQA 32:4)
- 128 experts per MoE layer, top-8 active per token
- Quantization: UD-Q4_K_XL via [Unsloth](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) (Dynamic 2.0)
- Disk: ~17.7 GB | Min VRAM: 16 GB (with expert offloading) | Max context: 262K native
- Built-in tool calling support via `--jinja` chat template

| GPU | VRAM | Context | Decode | 3-concurrent | VRAM used |
|-----|------|---------|--------|--------------|-----------|
| RTX 5090 | 32 GB | 64K/slot | ~275 tok/s | ~497 tok/s | 28.9 GB |
| Other NVIDIA (16 GB+) | 16+ GB | 16K/slot | varies | varies | varies |

<details>
<summary>RTX 5090 detailed benchmark</summary>

```
SINGLE-STREAM DECODE:    ~275 tok/s
3-CONCURRENT AGGREGATE:  ~497 tok/s  (+81% via MoE expert batching)
3-CONCURRENT PER-SLOT:   ~168 tok/s  each
PROMPT PROCESSING:       ~345-1,038 tok/s  (varies with batch position)
VRAM USAGE:              28.9 GB / 32.6 GB (3.7 GB headroom)
CONTEXT:                 64K per slot (3 slots, auto-fitted from 192K request)
```

Benchmarked 2026-03-02 with native sm_120a (Blackwell) compilation and `BLACKWELL_NATIVE_FP4=1` enabled.

**Why 3 slots (not 4)?** With 3 slots, `--fit on` allocates 64K context per slot instead of 48K. Aggregate throughput is identical (497 vs 495 tok/s), but per-agent speed under load is 35% faster (168 vs 124 tok/s). The 4th slot rarely matters for a single-GPU workstation. Override with `FOUNDRY_EXTRA_ARGS="--parallel 4"` if needed.

**vs Qwen3.5-35B-A3B**: 52% faster single-stream, 55% faster aggregate. The standard MoE architecture (no DeltaNet recurrent layers) batches more efficiently on Blackwell. Trades the 192K effective context of Qwen3.5 for raw speed.
</details>

## How It Works

Why llama.cpp and not SGLang or vLLM? For **consumer GPUs**, llama.cpp's MoE expert offloading (`--fit on`) is the only engine that can run a 35B-parameter MoE model on a single 16-24 GB card at full speed. SGLang and vLLM require the entire model to fit in VRAM.

Qwen3.5-35B-A3B keeps attention layers on GPU while spilling inactive experts to CPU, which is why a 35B MoE runs **faster** than a 27B dense model on the same hardware.

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

Available profiles (per model): `rtx5090`, `default`

### Architecture-Aware Tuning

The entrypoint automatically applies architecture-specific flags based on the `FOUNDRY_ARCH` environment variable baked into each image:

| Architecture | Flag | Reason |
|---|---|---|
| MoE (`moe`) | `--fit on` | Spill inactive experts to CPU when VRAM is tight |
| Dense (`dense`) | (none) | No experts to offload |

Model-specific quirks (e.g. `--swa-full` for Qwen's hybrid attention, `--cache-ram 0` for recurrent state) are set in profile `EXTRA_ARGS`, not in the architecture tier.

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

The RTX 5090 profiles are configured with multiple concurrent inference slots: `--parallel 4` for Qwen3.5 and Hermes, `--parallel 3` for Qwen3-Coder. This makes Foundry well-suited for multi-agent workflows where several AI agents share a single GPU.

### Why MoE batching works

Qwen3.5-35B-A3B uses a 256-expert MoE architecture with only 8 experts active per token. During single-stream decode, the GPU's tensor cores are largely idle -- the bottleneck is memory bandwidth, not compute. When multiple agents send concurrent requests, llama.cpp batches token generation across all active slots. Different tokens may route to different experts, and CUDA graphs capture the entire batched MoE operation, significantly improving GPU utilization.

### Throughput scaling

Measured on RTX 5090 with Qwen models (MoE):

| Active agents | Qwen3.5-35B-A3B (4 slots) | Qwen3-Coder-30B-A3B (3 slots) |
|---------------|----------------------------|--------------------------------|
| 1 | 181 tok/s | 275 tok/s |
| 2 | 234 tok/s (117 each) | 405 tok/s (204 each) |
| 3 | — | 497 tok/s (168 each) |
| 4 | 320 tok/s (80 each) | — |

Single-agent speed is unaffected. Concurrent slots only activate when there are simultaneous requests.

### Multi-GPU scaling

With 2x RTX 5090, run two independent instances for double the concurrent slots and aggregate throughput:

```bash
# GPU 0
docker run --gpus '"device=0"' -p 8080:8080 -v ~/.cache/foundry:/models \
  ghcr.io/infernet-org/foundry/qwen3-coder-30b-a3b:latest

# GPU 1
docker run --gpus '"device=1"' -p 8081:8080 -v ~/.cache/foundry:/models \
  ghcr.io/infernet-org/foundry/qwen3-coder-30b-a3b:latest
```

### Compatible frameworks

Any OpenAI-compatible agent framework works out of the box -- point it at `http://localhost:8080/v1`. See [AGENTS.md](AGENTS.md) for setup examples.

## Running

### Docker Compose

```bash
# Default: Qwen3.5-35B-A3B
docker compose up

# Choose a different model
FOUNDRY_MODEL=hermes-4.3-36b docker compose up
FOUNDRY_MODEL=qwen3-coder-30b-a3b docker compose up

# With explicit profile
FOUNDRY_PROFILE=rtx5090 docker compose up

# With monitoring stack (Prometheus + Grafana + GPU + eBPF metrics)
docker compose --profile monitoring up
```

Create a `.env` file for secrets and optional config:

```
HF_TOKEN=hf_your_token_here
GF_ADMIN_USER=admin
GF_ADMIN_PASSWORD=admin
```

### Build From Source

```bash
make build                        # Build the default model image (qwen3.5-35b-a3b)
make build MODEL=hermes-4.3-36b   # Build a different model
make build MODEL=qwen3-coder-30b-a3b  # Build the coding-optimized model
make run                          # Run with auto-detected GPU
make test                         # Smoke test: start, wait for health, send one request
make benchmark                    # Run benchmark against a running server
make download                     # Download the GGUF model file to ~/.cache/foundry
```

### Run Benchmark

```bash
python3 scripts/benchmark.py --url http://localhost:8080 --mode all
```

Modes: `all`, `generation` (single-stream decode), `prompt` (prompt processing), `throughput` (4-concurrent).

## Monitoring

Foundry includes an optional observability stack activated via Docker Compose profiles.

```bash
docker compose --profile monitoring up
```

### Stack Components

| Component | Port | Source | Metrics |
|-----------|------|--------|---------|
| **llama-server** | 8080 | Built-in `/metrics` | Decode tok/s, prompt tok/s, active slots, deferred requests, KV cache usage |
| **Prometheus** | 9090 | Scrapes all targets | Time-series storage, 30-day retention |
| **Grafana** | 3000 | Dashboards | Visualization (default: admin / admin) |
| **nvidia-gpu-exporter** | 9835 | `nvidia-smi` | VRAM, GPU utilization, temperature, power, clocks, fan speed |
| **node-exporter** | 9100 | `/proc`, `/sys` | CPU, RAM, disk, network, load average |
| **cAdvisor** | 8081 | Docker API | Per-container CPU, memory, network I/O |
| **ebpf-exporter** | 9435 | eBPF / kernel tracepoints | Block I/O latency histograms, scheduling latency, kernel-level metrics |

The eBPF exporter runs with `privileged: true` and `pid: host` to attach kernel tracepoints. It ships with Cloudflare's `biolatency` config by default, providing block I/O latency distributions useful for diagnosing model loading stalls and NVMe performance.

### Dashboards

All dashboards are auto-provisioned on first start -- no manual import needed.

| Dashboard | Description |
|-----------|-------------|
| **Foundry Inference** | Custom: inference throughput gauges, slot utilization, GPU telemetry, host resources |
| **Node Exporter Full** | Host metrics (community dashboard #1860) |
| **NVIDIA GPU** | GPU monitoring (community dashboard #14574) |
| **cAdvisor** | Container resources (community dashboard #14282) |

### Monitoring Architecture

```
┌─────────────────┐     ┌────────────────┐     ┌─────────┐
│  llama-server    │────▶│                │     │         │
│  :8080/metrics   │     │                │     │         │
├─────────────────┤     │                │     │         │
│  nvidia-gpu-exp  │────▶│  Prometheus    │────▶│ Grafana │
│  :9835           │     │  :9090         │     │ :3000   │
├─────────────────┤     │                │     │         │
│  node-exporter   │────▶│  scrapes 15s   │     │         │
│  :9100           │     │  30d retention │     │         │
├─────────────────┤     │                │     │         │
│  cAdvisor        │────▶│                │     │         │
│  :8081           │     │                │     │         │
├─────────────────┤     │                │     └─────────┘
│  ebpf-exporter   │────▶│                │
│  :9435           │     └────────────────┘
└─────────────────┘
```

## Host Kernel Tuning

For maximum performance, run the host tuning script once on the Docker host:

```bash
sudo ./scripts/host-setup.sh
```

Changes are **not persistent** across reboots. The script prints instructions for making them permanent via `/etc/sysctl.d/` and GRUB.

### What Gets Tuned

| Category | Parameter | Value | Purpose |
|----------|-----------|-------|---------|
| **Memory** | `vm.swappiness` | 0 | Keep model weights strictly in RAM |
| | `vm.overcommit_memory` | 1 | Ensure `mlock()` succeeds for large models |
| | `vm.nr_hugepages` | 1280 | ~2.5 GB hugepages for reduced TLB misses |
| | `kernel.numa_balancing` | 0 | Disable page migration jitter |
| | THP defrag | `defer+madvise` | Prevent allocation stalls |
| | `vm.dirty_ratio` | 80 | Reduce I/O contention during model load |
| **Network** | TCP congestion | BBR | Smoother token streaming over WAN |
| | `net.core.somaxconn` | 4096 | Handle connection bursts |
| | `net.core.busy_read/poll` | 50 us | Reduce NIC-to-CPU interrupt latency |
| | TCP fast open | Enabled | Faster connection setup |
| | Buffer sizes | 16 MB | Adequate for streaming responses |
| **I/O** | NVMe scheduler | `none` | NVMe handles its own queues |
| | NVMe read-ahead | 4 MB | Fast sequential model loading |
| **PCIe** | ASPM | Disabled | Prevent link sleep latency for MoE routing |
| **CPU** | Governor | `performance` | Maximum clock speed, no frequency scaling |
| | EPB | 0 | Maximum performance energy bias |
| **GPU** | Persistence mode | Enabled | Avoid ~100-500 ms cold start latency |

### GPU IRQ Pinning (Advanced)

For tighter tail latency, pin GPU interrupts to dedicated cores away from inference threads:

```bash
# Pin NVIDIA IRQs to cores 28-31 (adjust for your topology)
for irq in $(grep nvidia /proc/interrupts | awk '{print $1}' | tr -d ':'); do
  echo 28-31 > /proc/irq/$irq/smp_affinity_list
done
```

This reduced p99 latency jitter from ~5.8 tok/s spread to ~2.2 tok/s spread in our RTX 5090 testing. Average throughput is unchanged -- the benefit is consistency.

## Project Structure

```
foundry/
├── models/
│   ├── qwen3.5-35b-a3b/
│   │   ├── Dockerfile               # Multi-stage: compiles llama.cpp for sm_89 + sm_120a
│   │   ├── entrypoint.sh            # Copied from scripts/entrypoint.sh at build time
│   │   └── profiles/
│   │       ├── rtx5090.sh           # 192K ctx, 4 slots, ~320 tok/s aggregate
│   │       └── default.sh           # 16K ctx, conservative settings
│   ├── hermes-4.3-36b/
│   │   ├── Dockerfile               # Multi-stage: compiles llama.cpp for sm_89 + sm_120a
│   │   ├── entrypoint.sh            # Copied from scripts/entrypoint.sh at build time
│   │   └── profiles/
│   │       ├── rtx5090.sh           # 32K ctx, 4 slots, ~132 tok/s aggregate
│   │       └── default.sh           # 8K ctx, 24 GB minimum
│   └── qwen3-coder-30b-a3b/
│       ├── Dockerfile               # Multi-stage: compiles llama.cpp for sm_89 + sm_120a
│       ├── entrypoint.sh            # Copied from scripts/entrypoint.sh at build time
│       └── profiles/
│           ├── rtx5090.sh           # 192K ctx, 3 slots, ~497 tok/s aggregate
│           └── default.sh           # 32K ctx, conservative settings
├── scripts/
│   ├── entrypoint.sh                # Shared entrypoint (GPU detect, profile load, model download)
│   ├── benchmark.py                 # Generation speed, prompt processing, throughput
│   ├── optimize_5090.py             # Multi-config A/B testing harness
│   ├── download-model.sh            # Download GGUF outside Docker
│   └── host-setup.sh               # Linux kernel tuning for inference
├── monitoring/
│   ├── prometheus/prometheus.yml    # Scrape config (llama-server, GPU, node, cAdvisor, eBPF)
│   └── grafana/
│       ├── dashboards/              # 4 pre-provisioned dashboards (JSON)
│       └── provisioning/            # Datasource and dashboard auto-provisioning
├── docker-compose.yml               # Inference + monitoring stack (with eBPF exporter)
├── Makefile                         # build, run, test, benchmark, download
├── AGENTS.md                        # AI agent integration guide
└── .github/workflows/
    ├── build.yml                    # CI: build and push Docker images to GHCR
    └── lint.yml                     # CI: ruff (Python) + shellcheck (Bash)
```

## License

Apache-2.0
