#!/bin/bash
# ==============================================================================
# FAP Gate 4b: Agentic SWE capability (SWE-bench Verified via mini-SWE-agent)
# ==============================================================================
# Token-efficiency trick: litellm registry prices output at 1e-6/token, so
# instance_cost * 1e6 = exact output tokens per task.
# Usage: ./run-swebench.sh <slice e.g. 0:50> [output-dir]
set -euo pipefail
SLICE="${1:?usage: run-swebench.sh <slice> [output-dir]}"
OUT="${2:-results/swebench-${SLICE/:/-}}"
VENV="${FAP_EVAL_HOME:-$HOME/.cache/foundry/eval}/venv"
REG="${FAP_EVAL_HOME:-$HOME/.cache/foundry/eval}/litellm-registry.json"
BASE_URL="${FAP_BASE_URL:-http://localhost:8080/v1}"
MODEL="${FAP_MODEL:-hosted_vllm/qwen3.6-35b-a3b-nvfp4}"
LITELLM_MODEL_REGISTRY_PATH="$REG" MSWEA_COST_TRACKING=ignore_errors \
"$VENV/bin/mini-extra" swebench \
  --subset verified --split test --shuffle --slice "$SLICE" \
  -m "$MODEL" \
  -c swebench.yaml \
  -c "model.model_kwargs.api_base=$BASE_URL" \
  -c "model.model_kwargs.api_key=dummy" \
  -w 4 -o "$OUT"
