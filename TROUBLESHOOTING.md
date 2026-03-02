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
curl -s http://localhost:8080/metrics | grep -E "llama_server_(slots|ctx|model)"
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

### "CUDA error: no kernel image is available for execution on the device"

**Symptom:** Server crashes immediately after loading model.

**Cause:** The compiled CUDA kernels don't match your GPU architecture. Foundry images are built for sm_89 (Ada/RTX 40xx) and sm_120a (Blackwell/RTX 50xx).

**Fix:** If you have an older GPU (e.g. Ampere/RTX 30xx), rebuild the image with your architecture:
```bash
# In the Dockerfile, change CMAKE_CUDA_ARCHITECTURES:
-DCMAKE_CUDA_ARCHITECTURES="86;89;120a"   # Add 86 for Ampere
```

**Verify:** `docker compose logs inference | grep "CUDA"` should show successful backend loading.

---

## Out of Memory

### Container exits with "OOMKilled"

**Symptom:** Container disappears. `docker inspect` shows OOMKilled.

**Cause:** VRAM or system RAM exhausted. The RTX 5090 profile uses 1M total context (262K/slot x 4), which requires ~29.5 GB VRAM for Qwen3.5-9B.

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
2. Reduce parallel slots:
   ```bash
   FOUNDRY_EXTRA_ARGS="--parallel 1" docker compose up
   ```
3. Use more aggressive KV cache quantization (add to `FOUNDRY_EXTRA_ARGS`):
   ```bash
   FOUNDRY_EXTRA_ARGS="-ctk q4_0 -ctv q4_0" docker compose up
   ```

**Verify:** `nvidia-smi` should show VRAM usage within your GPU's capacity.

### "CUDA out of memory" in logs

**Symptom:** Server starts but crashes when processing the first request.

**Cause:** KV cache allocation exceeds available VRAM. This happens when context is too large for your GPU.

**Fix:** Same as OOMKilled above. Start with a small context and increase:
```bash
FOUNDRY_CTX_LENGTH=16384 docker compose up
```

---

## Startup & Model Loading

### "Download failed: xxx.gguf not found after download"

**Symptom:** First startup fails during model download.

**Cause:** Hugging Face API rate limiting, network issues, or missing auth token for gated models.

**Fix:**
1. Set a Hugging Face token (free, read-only access):
   ```bash
   echo "HF_TOKEN=hf_your_token_here" > .env
   docker compose up
   ```
2. Or download manually and mount:
   ```bash
   pip install huggingface-hub
   huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir ~/.cache/foundry
   docker compose up
   ```

**Verify:** `ls -lh ~/.cache/foundry/*.gguf` should show the model file.

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

### HTTP 503 / All slots busy

**Symptom:** API returns 503 when all parallel slots are occupied.

**Cause:** More concurrent requests than available slots. Default: 4 slots (RTX 5090) or 2 slots (default profile).

**Fix:**
1. Check slot usage:
   ```bash
   curl -s http://localhost:8080/metrics | grep "llama_server_requests"
   ```
2. Increase parallel slots (requires more VRAM):
   ```bash
   FOUNDRY_EXTRA_ARGS="--parallel 8" docker compose up
   ```
3. Or queue requests client-side with retry logic.

---

## Performance

### Throughput below documented benchmarks

**Symptom:** Decode speed is 50%+ lower than documented (e.g. 90 tok/s instead of 177 tok/s for Qwen3.5-9B).

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
1. Reduce concurrent load (fewer parallel slots = less heat):
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

**Cause:** The `@ai-sdk/openai` package includes `extractReasoningMiddleware` that crashes when it encounters `<think>` tokens in the response content. Qwen3.5 models always generate `<think>` blocks even with thinking disabled ([vercel/ai #12054](https://github.com/vercel/ai/issues/12054)).

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

Also ensure the server uses `--reasoning-format none` (already set in Foundry's Qwen3.5-9B profiles).

### OpenCode: Tool calls appear as raw XML

**Symptom:** Model outputs `<tool_call><function=...>` as text instead of executing tools.

**Cause:** The server is returning `reasoning_content` in the API response, which confuses the AI SDK's tool call parser.

**Fix:** Ensure `--reasoning-format none` is in the server's `PROFILE_EXTRA_ARGS`. This is the default in Foundry's Qwen3.5-9B profiles. If you're using a custom profile or `FOUNDRY_EXTRA_ARGS`, add it:
```bash
FOUNDRY_EXTRA_ARGS="--reasoning-format none" docker compose up
```

**Verify:**
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-9B-UD-Q4_K_XL.gguf","messages":[{"role":"user","content":"hello"}],"max_tokens":64}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('reasoning_content' in d['choices'][0]['message'])"
# Should print: False
```

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
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool | grep -E '"health"|"lastError"'
```

**Fix:**
1. Wait 30-60 seconds after starting services (Prometheus scrapes every 15s).
2. Verify the inference metrics endpoint works:
   ```bash
   curl -s http://localhost:8080/metrics | head -5
   ```
3. Check that `monitoring/prometheus/prometheus.yml` has correct scrape targets.

**Verify:** Prometheus targets page at `http://localhost:9090/targets` should show all targets as "UP".

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
