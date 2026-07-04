#!/bin/bash
# ==============================================================================
# FAP Gate 2: Deployment fidelity (EvalPlus HumanEval+, greedy)
# ==============================================================================
# Usage: ./run-evalplus.sh <tag> [base_url] [model]
set -euo pipefail
TAG="${1:?usage: run-evalplus.sh <tag> [base_url] [model]}"
BASE_URL="${2:-http://localhost:8080/v1}"
MODEL="${3:-qwen3.6-35b-a3b-nvfp4}"
VENV="${FAP_EVAL_HOME:-$HOME/.cache/foundry/eval}/venv"
[ -x "$VENV/bin/python" ] || { echo "venv missing: pip install evalplus into $VENV"; exit 1; }
mkdir -p results
export OPENAI_API_KEY=dummy
"$VENV/bin/python" -m evalplus.evaluate \
  --model "$MODEL" --dataset humaneval --backend openai \
  --base-url "$BASE_URL" --greedy \
  --root "results/evalplus-${TAG}" 2>&1 | tee "results/evalplus-${TAG}.log"
