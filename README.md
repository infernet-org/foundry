# Foundry

Tuned Docker image for running Qwen3.6-35B-A3B-NVFP4 on consumer Blackwell GPUs. One command, maximum tok/s.

Foundry serves NVIDIA's [ModelOpt NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) with [vLLM](https://github.com/vllm-project/vllm), bundles per-GPU hardware profiles, and auto-detects your GPU at startup. No manual tuning required.

**Requires an NVFP4-capable GPU**: Blackwell (RTX 50xx, sm_120) or Hopper (sm_90), with 32 GB+ VRAM.

## Table of Contents

- [Quick Start](#quick-start)
- [FAP: The Foundry Assessment Protocol](#fap-the-foundry-assessment-protocol)
- [Models](#models)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Multi-Agent Inference](#multi-agent-inference)
- [Running](#running)
- [Monitoring](#monitoring)
- [AI Agents & Skills](#ai-agents--skills)
- [Host Kernel Tuning](#host-kernel-tuning)
- [Project Structure](#project-structure)
- [License](#license)

## Quick Start

```bash
docker run --gpus all --shm-size 8g -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  ghcr.io/infernet-org/foundry/qwen3.6-35b-a3b-nvfp4:latest
```

The first run downloads the model (~22 GB). Subsequent starts take 2-4 minutes (weight loading + CUDA graph capture).

Then use it like any OpenAI-compatible API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b-nvfp4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Works with any OpenAI-compatible client: Cursor, Continue, OpenCode, Open WebUI, CrewAI, AutoGen, etc. See [AGENTS.md](AGENTS.md) for detailed integration guides.

## FAP: The Foundry Assessment Protocol

Every shipped configuration passes four gates before it earns a profile:

| Gate | Question | This deployment |
|------|----------|-----------------|
| 1. Throughput ceiling | Fastest correct config? | **384 tok/s single / 1,228 tok/s 4-concurrent** |
| 2. Deployment fidelity | Does our stack corrupt output? | HumanEval+ **88.4%** greedy -- PASS |
| 3. Quant preservation | Did NVFP4 hurt the model? | **>=99.2%** of BF16 on all suites |
| 4. Measured intelligence | Real SWE capability? | Aider polyglot **50.2%** pass@2 (thinking off) |

Methodology, full sweep record, and runners: **[EVALUATION.md](EVALUATION.md)** + `scripts/eval/`.

## Models

### Qwen3.6-35B-A3B-NVFP4 (MoE)

Hybrid Gated DeltaNet + MoE, Qwen3.6 generation. 35B total parameters, ~3B active per token. Served by vLLM: the checkpoint is NVIDIA ModelOpt NVFP4 (4-bit floating point safetensors), a format llama.cpp cannot load.

- 40 layers, hybrid recurrent + full attention, MoE experts
- Quantization: NVFP4 language model + BF16 vision tower + FP8 KV cache, via [NVIDIA ModelOpt](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
- Disk: ~22 GB | Min VRAM: 32 GB | Max context: 262K native
- **Requires Hopper (sm_90) or Blackwell (RTX 50xx) GPU** -- NVFP4 does not run on Ada or older
- Thinking mode via `reasoning_content` (qwen3 reasoning parser)
- Vision input supported by the checkpoint but disabled by default to save VRAM (`PROFILE_MULTIMODAL=true` to enable)

| GPU | VRAM | Context | Decode | 4-concurrent | VRAM used |
|-----|------|---------|--------|--------------|-----------|
| RTX 5090 | 32 GB | 224K | ~384 tok/s | ~1,228 tok/s | 29.0 GB |
| Other NVFP4-capable (32 GB+) | 32+ GB | 32K | varies | varies | varies |

The RTX 5090 numbers use **MTP x4 self-speculative decoding** (the checkpoint ships its own draft head) + async scheduling -- 1.9x single-stream and 2.3x concurrent over the plain configuration. To trade speed for the full 262K context, edit `PROFILE_EXTRA_ARGS` in the profile (drop `--speculative-config`) and raise `PROFILE_CTX_LENGTH` -- `FOUNDRY_EXTRA_ARGS` appends flags and cannot remove them.

Sweep record and per-config numbers: [EVALUATION.md](EVALUATION.md).

## How It Works

### GPU Auto-Detection

On startup, Foundry:
1. Detects your GPU via `nvidia-smi` and verifies NVFP4 capability (compute capability >= 9.0)
2. Loads a tuned hardware profile with optimal settings
3. Downloads the model snapshot if not already cached
4. Launches `vllm serve` with the right arguments

### Hardware Profiles

Each profile tunes: context length (`--max-model-len`), VRAM budget (`--gpu-memory-utilization`), concurrency (`--max-num-seqs`), prefill chunking (`--max-num-batched-tokens`), and the MoE kernel backend.

```bash
# Override auto-detection with a specific profile
docker run --gpus all --shm-size 8g -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  -e FOUNDRY_PROFILE=rtx5090 \
  ghcr.io/infernet-org/foundry/qwen3.6-35b-a3b-nvfp4:latest
```

Available profiles: `rtx5090`, `default`

### NVFP4 on consumer Blackwell

The checkpoint stores the language model in NVFP4 (4-bit floating point with per-block FP8 scales) and the KV cache in FP8. On sm_120 vLLM auto-selects the MARLIN weight-only kernel (robust, no warmup). The native-FP4 `flashinfer_b12x` backend is ~4% faster at decode but experimental -- opt in via `PROFILE_MOE_BACKEND=flashinfer_b12x` in the profile or `FOUNDRY_EXTRA_ARGS="--moe-backend flashinfer_b12x"`.

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FOUNDRY_PROFILE` | `auto` | GPU profile (`auto`, `rtx5090`, `default`) |
| `FOUNDRY_PORT` | `8080` | Server port |
| `FOUNDRY_CTX_LENGTH` | Profile default | Context window size (`--max-model-len`) |
| `FOUNDRY_EXTRA_ARGS` | (empty) | Additional `vllm serve` arguments (highest priority) |
| `HF_TOKEN` | (empty) | Hugging Face token for authenticated downloads |

## Multi-Agent Inference

vLLM's continuous batching schedules concurrent requests dynamically -- no fixed slot count. The RTX 5090 profile allows up to 8 concurrent sequences (`--max-num-seqs 8`), making Foundry well-suited for multi-agent workflows sharing a single GPU.

### Why MoE batching works

Only ~3B of 35B parameters activate per token. During single-stream decode the GPU is memory-bandwidth-bound; tensor cores sit mostly idle. Concurrent requests batch across sequences -- different tokens route to different experts -- multiplying aggregate throughput without hurting per-stream speed much.

### Throughput scaling

Measured on RTX 5090 (vLLM 0.24.0, MARLIN backend):

| Active agents | Aggregate | Per-agent |
|---------------|-----------|-----------|
| 1 | ~384 tok/s | ~384 tok/s |
| 4 | ~1,228 tok/s | ~307 tok/s |

### Multi-GPU scaling

With 2x RTX 5090, run two independent instances:

```bash
# GPU 0
docker run --gpus '"device=0"' --shm-size 8g -p 8080:8080 -v ~/.cache/foundry:/models \
  ghcr.io/infernet-org/foundry/qwen3.6-35b-a3b-nvfp4:latest

# GPU 1
docker run --gpus '"device=1"' --shm-size 8g -p 8081:8080 -v ~/.cache/foundry:/models \
  ghcr.io/infernet-org/foundry/qwen3.6-35b-a3b-nvfp4:latest
```

### Compatible frameworks

Any OpenAI-compatible agent framework works out of the box -- point it at `http://localhost:8080/v1`. See [AGENTS.md](AGENTS.md) for setup examples.

## Running

### Docker Compose

```bash
docker compose up

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
make build      # Build the model image
make run        # Run with auto-detected GPU
make test       # Smoke test: start, wait for health, send one request
make benchmark  # Run benchmark against a running server
make download   # Download the model weights (~22 GB) to ~/.cache/foundry
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
| **vLLM** | 8080 | Built-in `/metrics` | Decode/prefill tok/s, running/waiting requests, KV cache usage, TTFT/TPOT histograms |
| **Prometheus** | 9091 (host; `PROM_PORT` to change) | Scrapes all targets | Time-series storage, 30-day retention |
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
| **Foundry Inference (vLLM)** | Custom: throughput gauges, MTP acceptance, latency percentiles (TTFT/ITL), KV cache, GPU telemetry |
| **Node Exporter Full** | Host metrics (community dashboard #1860) |
| **NVIDIA GPU** | GPU monitoring (community dashboard #14574) |
| **cAdvisor** | Container resources (community dashboard #14282) |

### Monitoring Architecture

```
┌─────────────────┐     ┌────────────────┐     ┌─────────┐
│  vLLM server     │────▶│                │     │         │
│  :8080/metrics   │     │                │     │         │
├─────────────────┤     │                │     │         │
│  nvidia-gpu-exp  │────▶│  Prometheus    │────▶│ Grafana │
│  :9835           │     │  :9091 (host)  │     │ :3000   │
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

## AI Agents & Skills

Foundry is agent-friendly in both directions:

- **Agents using the API**: any OpenAI-compatible framework — see [AGENTS.md](AGENTS.md).
- **Agents working on this repo**: [CLAUDE.md](CLAUDE.md) is the meta-index
  (repo map, commands, operational rules). Read by Claude Code natively;
  Codex and others are routed there from AGENTS.md.
- **Skills** (Claude Code, Codex, and other skills-compatible agents):

```bash
npx skills add infernet-org/foundry
```

| Skill | What it does |
|-------|--------------|
| `foundry-serve` | Start/stop/health-check the inference server and monitoring stack |
| `foundry-benchmark` | Throughput + MTP-acceptance measurement against a running server |
| `foundry-assess` | Run the FAP quality gates (fidelity, coding, agentic SWE) |

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
│   └── qwen3.6-35b-a3b-nvfp4/
│       ├── Dockerfile               # vLLM backend (NVFP4 -- Blackwell/Hopper only)
│       ├── entrypoint.sh            # GPU detect, profile load, snapshot download, vllm serve
│       └── profiles/
│           ├── rtx5090.sh           # 224K ctx, MTP x4, ~1,228 tok/s aggregate
│           └── default.sh           # 32K ctx, 32 GB minimum
├── scripts/
│   ├── benchmark.py                 # Generation speed, prompt processing, throughput
│   ├── eval/                        # FAP gate runners (evalplus, aider, swebench)
│   ├── download-model.sh            # Download model weights outside Docker
│   └── host-setup.sh               # Linux kernel tuning for inference
├── skills/                          # Agent skills (npx skills add infernet-org/foundry)
├── CLAUDE.md                        # Agent meta-index (repo map, commands, rules)
├── EVALUATION.md                    # FAP certification record
├── monitoring/
│   ├── prometheus/prometheus.yml    # Scrape config (vLLM, GPU, node, cAdvisor, eBPF)
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
