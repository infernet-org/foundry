---
name: foundry-serve
description: Start, stop, and health-check the foundry vLLM inference server (Qwen3.6-35B-A3B-NVFP4). Use when asked to serve the model, restart inference, check why the server is down, or bring up the monitoring stack.
---

# Serving foundry

## Start
```bash
make run                                    # foreground, auto GPU profile
docker compose up -d                        # detached
docker compose --profile monitoring up -d   # + Prometheus (:9091) / Grafana (:3000)
```
First run downloads ~22 GB; every start needs 2-4 min (weight load + CUDA graph capture).
Do not declare failure before 5 minutes — poll `curl -sf localhost:8080/health`.

## Verify
```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b-nvfp4","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

## Diagnose
- Container exited: `docker logs <container> 2>&1 | grep -iE "error|OOM" | tail`
- KV/OOM at startup → lower `FOUNDRY_CTX_LENGTH` (e.g. 131072); never raise gpu-memory-utilization past 0.90 on 32 GB
- GPU too old: needs Blackwell (RTX 50xx) or Hopper — the entrypoint fails fast with the reason
- Full playbook: TROUBLESHOOTING.md

## Stop
```bash
docker compose --profile monitoring down    # or docker rm -f <container>
```
