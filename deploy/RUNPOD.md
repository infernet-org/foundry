# Foundry on RunPod Serverless

Deploy Foundry as a serverless GPU inference endpoint on RunPod.

## Quick Deploy

1. **Push to GitHub** (triggers CI to build images)
   ```bash
   git push origin main
   ```

2. **Wait for CI to complete** (~10-15 minutes)
   - Check: https://github.com/infernet-org/foundry/actions

3. **Create RunPod Template** with the   - Image: `ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:latest`
   - GPU: RTX 4090, A100, or H100
   - Memory: 32GB+
   - Port: 8080

4. **Deploy Serverless Endpoint**
   - **Crucial**: You must create a **Load Balancing Endpoint** (not a Queue-Based Endpoint).
   - Select the template
   - Set min/max workers
   - Ensure both 8080 and 8081 are exposed in the configuration
   - Get your endpoint URL

## Architecture

```
                         RunPod Serverless
                                │
                    ┌───────┴───────┐
                    │                 │
              ┌─────┴─────┐     ┌─────┴─────┐
              │  Worker 1  │     │  Worker 2  │
              │  (GPU)     │     │  (GPU)     │
              │  llama.cpp │     │  llama.cpp │
              └───────────┘     └───────────┘
                    │                 │
                    └───────┬───────┘
                            │
                    Load Balancer
                            │
                      Client API
```

## Endpoint URL Format

Once deployed, your endpoint will be available at:

```
https://your-endpoint-id.runpod.net/v1/chat/completions
```

## Example Usage

```python
import openai

client = openai.OpenAI(
    base_url="https://your-endpoint-id.runpod.net/v1",
    api_key="anything"  # RunPod doesn't require real API key
)

response = client.chat.completions.create(
    model="qwen3.5-35b-a3b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Cost Optimization

RunPod Serverless only charges for active compute time:

| Workers | GPU      | Rate         | Idle Cost |
|---------|----------|--------------|-----------|
| 0       | Any      | $0.00/hr     | Free      |
| 1       | RTX 4090 | ~$0.20/hr    | $0.00      |
| 1       | A100     | ~$1.50/hr    | $0.00      |

**Tip**: Set `min_workers: 0` to scale to zero during idle periods.

## Cold Start Optimization

The model (~20GB) is downloaded on first container start. To speed up cold starts:

### Option 1: Pre-download during build (Slower build, faster start)

Set `PREBAKE_MODEL=true` in your RunPod environment variables.

### Option 2: Network Volume (Fastest)

Attach a RunPod Network Volume to cache the model across worker restarts:

```yaml
volumes:
  - name: qwen-model-cache
    mount_path: /models
    size: 50Gi
```

## Scaling Configuration

| Parameter | Recommendation | Description |
|-----------|---------------|-------------|
| min_workers | 0 | Scale to zero when idle |
| max_workers | 10 | Maximum concurrent GPUs |
| idle_timeout | 300 | 5 min before scaling down |
| scale_up_delay | 30 | Wait before adding workers |

## Available GPU Types

| GPU       | VRAM   | Model Fits | Recommended For |
|-----------|--------|------------|-----------------|
| RTX 3090  | 24GB   | Yes (offload) | Development, testing |
| RTX 4090  | 24GB   | Yes (offload) | Production, single-user |
| RTX 5090  | 32GB   | Fully      | Production, high-throughput |
| A100      | 80GB   | Fully      | Production, multi-user |
| H100      | 80GB   | Fully      | Production, maximum throughput |

## Monitoring

Check worker health:
```bash
curl https://your-endpoint-id.runpod.net/health
```

View logs in RunPod dashboard:
- Worker startup logs
- Inference requests
- Memory usage
- GPU utilization

## Troubleshooting

### Slow Cold Starts
- Pre-download model during image build
- Use a Network Volume for model caching
- Increase `scale_up_delay` to allow more startup time

### OOM Errors
- Reduce `FOUNDRY_CTX_LENGTH`
- Set `FOUNDRY_PROFILE=rtx4080` for smaller GPUs
- Reduce `FOUNDRY_PARALLEL`

### High Latency
- Check GPU type (A100/H100 faster than consumer GPUs)
- Reduce concurrent requests (`FOUNDRY_PARALLEL=1`)
- Check network volume performance

## GitHub CI/CD Integration

Your GitHub Actions workflow already pushes to GHCR. To auto-deploy to RunPod:

1. **Add RunPod API Token** to GitHub secrets: `RUNPOD_API_KEY`
2. **Use RunPod API** to update template on push:

```yaml
# .github/workflows/deploy-runpod.yml
- name: Update RunPod Template
  run: |
    curl -X PATCH "https://api.runpod.io/v1/template/your-template-id" \
      -H "Authorization: Bearer ${{ secrets.RUNPOD_API_KEY }}" \
      -H "Content-Type: application/json" \
      -d '{"image": "ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:${{ github.sha }}"}
```
