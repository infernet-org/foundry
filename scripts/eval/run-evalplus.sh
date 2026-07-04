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

# Preflight: evalplus caps generation at 768 tokens and cannot send
# chat_template_kwargs. With thinking enabled (the serving default) the
# budget is consumed inside <think> and content returns null, crashing the
# sanitizer. Gate 2 therefore requires a thinking-off serving config.
probe=$(curl -sf -m 60 "$BASE_URL/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":128}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")' 2>/dev/null)
if [ -z "$probe" ]; then
    echo "ERROR: the server returned empty content (thinking mode is consuming the token budget)." >&2
    echo "Gate 2 needs a thinking-off serving config. Restart with:" >&2
    echo "  FOUNDRY_EXTRA_ARGS='--default-chat-template-kwargs {\"enable_thinking\":false}' make run" >&2
    echo "then rerun this gate. (This tests the same quant/parser/serving chain; only the" >&2
    echo "chat-template default differs.)" >&2
    exit 1
fi
"$VENV/bin/python" -m evalplus.evaluate \
  --model "$MODEL" --dataset humaneval --backend openai \
  --base-url "$BASE_URL" --greedy \
  --root "results/evalplus-${TAG}" 2>&1 | tee "results/evalplus-${TAG}.log"
