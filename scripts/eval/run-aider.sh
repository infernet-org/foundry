#!/bin/bash
# ==============================================================================
# FAP Gate 3a: Coding capability + token efficiency (Aider polyglot, 225 tasks)
# ==============================================================================
# Requires: aider repo + polyglot-benchmark cloned, aider-benchmark image built
# (see EVALUATION.md). Rootless-docker aware: talks to the serving container
# by bridge IP instead of host-gateway.
# Usage: ./run-aider.sh <run-name> [extra benchmark.py args...]
set -euo pipefail
RUN="${1:?usage: run-aider.sh <run-name> [args...]}"; shift || true
AIDER_DIR="${FAP_EVAL_HOME:-$HOME/.cache/foundry/eval}/aider"
CONTAINER="${FAP_CONTAINER:-$(docker ps --format '{{.Names}}' | grep -E 'foundry|vllm' | head -1)}"
VIP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CONTAINER")
PORT="${FAP_CONTAINER_PORT:-8080}"   # foundry serves on 8080 in-container (FOUNDRY_PORT)
[ -n "$VIP" ] || { echo "serving container not found"; exit 1; }
cd "$AIDER_DIR"
docker run --rm \
  -v "$PWD":/aider -v "$PWD/tmp.benchmarks":/benchmarks \
  -e OPENAI_API_BASE="http://${VIP}:${PORT}/v1" -e OPENAI_API_KEY=dummy \
  -e HISTFILE=/dev/null -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks \
  aider-benchmark \
  ./benchmark/benchmark.py "$RUN" \
    --model "openai/${FAP_MODEL:-qwen3.6-35b-a3b-nvfp4}" --edit-format diff --threads 4 \
    --exercises-dir polyglot-benchmark "$@"
