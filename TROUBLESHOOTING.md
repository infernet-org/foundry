# Troubleshooting

Common issues and solutions for Foundry inference servers.

## First Steps: Collecting Diagnostic Information

Before troubleshooting, gather this information:

### Container Logs

```bash
docker compose logs inference --tail 50
```

### Health Check

```bash
curl http://localhost:8080/health
# Returns: {"status":"ok"} when healthy
```

### Metrics

```bash
curl -s http://localhost:8080/metrics | grep -E "vllm:(num_requests|kv_cache|generation_tokens)"
```

### GPU Status

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu,persistence_mode --format=csv
```

---

## GPU & CUDA

### "no devices with dedicated memory found"

**Symptom:** Server starts but model loads on CPU. Inference is extremely slow.

**Cause:** The NVIDIA container runtime is not mounting GPU drivers into the container.

**Fix:**
1. Verify the host GPU works:
   ```bash
   nvidia-smi
   ```
2. Verify Docker can access the GPU:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
   ```
3. If step 2 fails, install the NVIDIA Container Toolkit:
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

**Verify:** `docker compose logs inference | grep "offloaded"` should show `offloaded N/N layers to GPU`.

### "CUDA error: no kernel image is available" / NVFP4 capability errors

**Cause:** the GPU is older than the NVFP4 checkpoint supports. This image requires
compute capability >= 9.0: Hopper (sm_90) or Blackwell (sm_100/sm_120, RTX 50xx).
The entrypoint fails fast with a clear message on older GPUs; if you see a raw
kernel-image error instead, you bypassed the entrypoint.

**Fix:** run on a supported GPU, or serve a GGUF build of this model with
llama.cpp on older hardware (outside the scope of this repo).

## Out of Memory

### Container exits with "OOMKilled"

**Symptom:** Container disappears. `docker inspect` shows OOMKilled.

**Cause:** VRAM or system RAM exhausted. The RTX 5090 profile runs 224K context with MTP speculative decoding at `--gpu-memory-utilization 0.90` (~29 GB VRAM).

**Diagnosis:**
```bash
docker inspect --format='{{.State.OOMKilled}}' foundry-inference-1
# Returns: true if OOM killed
```

**Fix:**
1. Reduce context length:
   ```bash
   FOUNDRY_CTX_LENGTH=65536 docker compose up
   ```
2. Reduce context or concurrency:

   ```bash
   FOUNDRY_CTX_LENGTH=131072 make run       # smaller KV allocation
   # or edit PROFILE_MAX_NUM_SEQS / PROFILE_GPU_MEM_UTIL in the profile, then `make build` (profiles are baked into the image)
   ```

### "CUDA out of memory" in logs

**Symptom:** Server starts but crashes when processing the first request.

**Cause:** KV cache allocation exceeds available VRAM. This happens when context is too large for your GPU.

**Fix:** Same as OOMKilled above. Start with a small context and increase:
```bash
FOUNDRY_CTX_LENGTH=16384 docker compose up
```

---

## Startup & Model Loading

### Download failed / model dir incomplete

**Symptom:** startup logs `Download failed` or vLLM crashes loading safetensors.

**Cause:** the ~22 GB snapshot download was interrupted. The entrypoint writes
`.foundry_download_complete` inside the model dir only after a full download and
resumes automatically on the next start; a crash mid-load usually means the
resume also failed (disk space, network, HF rate limits).

**Fix:**
1. Check disk space: `df -h ~/.cache/foundry` (need ~25 GB free).
2. Restart the container -- `snapshot_download` resumes incrementally.
3. Or pre-download outside docker: `./scripts/download-model.sh`
4. Private/gated repo? Pass `HF_TOKEN` via `.env`.

**Verify:** `du -sh ~/.cache/foundry/Qwen3.6-35B-A3B-NVFP4` shows ~22 GB and the
`.foundry_download_complete` marker exists.

### "Permission denied: Cannot access /models"

**Symptom:** Container starts but cannot read the model volume.

**Cause:** Docker volume permissions don't match the container user.

**Fix:**
```bash
sudo chown -R $(id -u):$(id -g) ~/.cache/foundry
chmod 755 ~/.cache/foundry
```

**Verify:** `docker compose logs inference | grep "Model found"` confirms the model is accessible.

### Model loading takes 30+ seconds

**Symptom:** Long delay between container start and first `/health` OK.

**Cause:** First-time model loading, GPU not in persistence mode, or slow storage.

**Fix:**
1. Enable GPU persistence mode (avoids ~100-500ms cold start per request):
   ```bash
   sudo nvidia-smi -pm 1
   ```
2. Run host tuning for optimized I/O:
   ```bash
   sudo ./scripts/host-setup.sh
   ```
3. Subsequent starts are fast since the model is cached in `~/.cache/foundry`.

**Verify:** `docker compose logs inference | grep "Model found"` shows the cached model with its size.

---

## API & Connectivity

### Connection refused on port 8080

**Symptom:** `curl: (7) Failed to connect to localhost port 8080: Connection refused`

**Cause:** Container is still loading the model, or another service is using port 8080.

**Fix:**
1. Check if the model is still loading:
   ```bash
   docker compose logs inference --tail 5
   # Look for "server is listening on" message
   ```
2. Check for port conflicts:
   ```bash
   ss -tlnp src :8080
   ```
3. Use a different port:
   ```bash
   FOUNDRY_PORT=8090 docker compose up
   ```

**Verify:** `curl http://localhost:8080/health` returns `{"status":"ok"}`.

### Requests queueing / slow under concurrency

**Symptom:** latency grows with many concurrent requests.

**Cause:** more concurrent requests than `--max-num-seqs` (default 8 in the
RTX 5090 profile); vLLM queues the excess rather than erroring.

**Fix:**
1. Check queue depth:

   ```bash
   curl -s http://localhost:8080/metrics | grep -E "vllm:num_requests_(running|waiting)"
   ```

2. Raise `PROFILE_MAX_NUM_SEQS` (costs KV/VRAM headroom) or add a second GPU/instance.

## Performance

### Throughput below documented benchmarks

**Symptom:** Decode speed is 50%+ lower than documented (e.g. 190 tok/s instead of ~384 tok/s single-stream).

**Cause:** Host kernel not tuned. The documented benchmarks assume `host-setup.sh` has been run.

**Diagnosis:**
```bash
# Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Should be: performance

# Check NUMA balancing
sysctl kernel.numa_balancing
# Should be: 0

# Check GPU persistence mode
nvidia-smi --query-gpu=persistence_mode --format=csv,noheader
# Should be: Enabled

# Check BBR congestion control (for streaming latency)
sysctl net.ipv4.tcp_congestion_control
# Should be: bbr
```

**Fix:** Run the host tuning script:
```bash
sudo ./scripts/host-setup.sh
```

This sets: CPU governor to performance, disables NUMA balancing, enables BBR TCP, tunes NVMe I/O, enables GPU persistence, allocates hugepages, and enables busy polling for reduced NIC latency.

**Verify:** Re-run your benchmark. Single-stream decode should be within 10% of documented speeds.

### Latency spikes after 10-15 minutes (thermal throttling)

**Symptom:** Steady performance degrades over time. GPU temp >85C.

**Diagnosis:**
```bash
nvidia-smi -q -d PERFORMANCE | grep -A5 "Clocks Throttle Reasons"
# Look for "SW Thermal Slowdown: Active"
```

**Fix:**
1. Reduce concurrent load (fewer concurrent sequences = less heat):
   ```bash
   FOUNDRY_EXTRA_ARGS="--parallel 2" docker compose up
   ```
2. Check `nvidia-smi dmon -s p` for real-time power/temp monitoring.

**Verify:** `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader` should stay below 83C under load.

---

## Docker & Container Issues

### Rootless Docker: sysctls and ulimits fail

**Symptom:** Container fails to start with `sysctl not allowed` or `permission denied setting ulimit`.

**Cause:** Rootless Docker cannot set privileged sysctls (`tcp_congestion_control`, `busy_read`, `busy_poll`) or unlimited memlock.

**Fix:** These settings are intentionally omitted from `docker-compose.yml` for rootless compatibility. Apply them at the host level instead:
```bash
sudo ./scripts/host-setup.sh
```

This is more effective anyway -- container-level network sysctls share the host's network stack.

### eBPF exporter won't start

**Symptom:** `ebpf-exporter` container exits immediately.

**Cause:** Requires `privileged: true` and `pid: host` for kernel tracepoint access. Incompatible with rootless Docker, SELinux enforcing, and some cloud providers.

**Fix:** The eBPF exporter is optional (part of the `monitoring` profile). The other monitoring services (Prometheus, Grafana, GPU exporter, node exporter) work without it.

To run without eBPF:
```bash
# The monitoring profile still works -- eBPF exporter will fail but others start fine
docker compose --profile monitoring up -d
```

**Verify:** `docker compose --profile monitoring ps` -- all services should be "Up" except `ebpf-exporter`.

---

## AI Agent Integration

### OpenCode: "text part msg_... not found"

**Symptom:** OpenCode crashes immediately with `text part msg_XXXX not found` when using `@ai-sdk/openai`.

**Cause:** The `@ai-sdk/openai` package includes `extractReasoningMiddleware` that crashes when it encounters `<think>` tokens in the response content. Thinking models emit `<think>` blocks; Foundry serves with `--reasoning-parser qwen3`, which moves reasoning into the separate `reasoning_content` field so message content stays clean ([vercel/ai #12054](https://github.com/vercel/ai/issues/12054)).

**Fix:** Use `@ai-sdk/openai-compatible` instead of `@ai-sdk/openai` in your `opencode.json`:
```json
{
  "provider": {
    "foundry": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "sk-local"
      }
    }
  }
}
```

Foundry already separates reasoning server-side via `--reasoning-parser qwen3`.

### OpenCode: Tool calls appear as raw XML

**Symptom:** Model outputs `<tool_call><function=...>` as text instead of executing tools.

**Cause:** The server is returning `reasoning_content` in the API response, which confuses the AI SDK's tool call parser.

**Fix:** the `qwen3` reasoning parser (set in the entrypoint) returns reasoning in
`reasoning_content`, separate from `content`. To disable thinking entirely, send
`"chat_template_kwargs": {"enable_thinking": false}` in the request, or serve with
`FOUNDRY_EXTRA_ARGS='--default-chat-template-kwargs {"enable_thinking":false}'`.

### Client requires API key

**Symptom:** Client errors with "No API key provided" even though Foundry doesn't require one.

**Fix:** Use any non-empty string as the API key:
```bash
export OPENAI_API_KEY=sk-local
```

---

## Monitoring

### Grafana dashboards show no data

**Symptom:** Dashboards load but all panels are empty.

**Cause:** Prometheus hasn't scraped targets yet, or targets are unreachable.

**Diagnosis:**
```bash
# Check Prometheus targets
curl -s http://localhost:9091/api/v1/targets | python3 -m json.tool | grep -E '"health"|"lastError"'
```

**Fix:**
1. Wait 30-60 seconds after starting services (Prometheus scrapes every 15s).
2. Verify the inference metrics endpoint works:
   ```bash
   curl -s http://localhost:8080/metrics | head -5
   ```
3. Check that `monitoring/prometheus/prometheus.yml` has correct scrape targets.

**Verify:** Prometheus targets page at `http://localhost:9091/targets` should show all targets as "UP".

---

## Collecting Information for Bug Reports

When filing an issue, include the output of these commands:

```bash
# System info
nvidia-smi
docker version
docker compose version
uname -a

# Container state
docker compose --profile monitoring ps
docker compose logs inference --tail 100

# Server health
curl -s http://localhost:8080/health
curl -s http://localhost:8080/v1/models | python3 -m json.tool

# Metrics snapshot
curl -s http://localhost:8080/metrics
```
