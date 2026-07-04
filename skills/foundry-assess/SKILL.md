---
name: foundry-assess
description: Run FAP (Foundry Assessment Protocol) quality gates — HumanEval+ fidelity gate, Aider polyglot, SWE-bench — against the served model. Use after any serving-config change, or when asked to evaluate/certify model quality or token efficiency.
---

# FAP — assessing the deployment

Full protocol + pass criteria: EVALUATION.md. One-time harness setup is at the
bottom of that file (venv + aider clone under `$FAP_EVAL_HOME`, default
`~/.cache/foundry/eval`).

## Gate 2 — fidelity (ALWAYS after config changes; ~2 min)
```bash
./scripts/eval/run-evalplus.sh mytag
```
Pass: HumanEval+ pass@1 in the 85-89% band. A drop means the serving stack
(quant kernels / reasoning parser / chat template / spec decode) corrupted output.

## Gate 4a — coding + token efficiency (~30 min - 2 h)
```bash
./scripts/eval/run-aider.sh myrun
```
Reference: 50.2% pass@2 thinking-off (~3.1K tok/case); 61.2% thinking-on (~13.4K tok/case).

## Gate 4b — agentic SWE (~2-3 h for a 50-slice)
```bash
./scripts/eval/run-swebench.sh 0:50
```
Tokens-per-solved-issue: instance_cost x 1e6 (registry prices output at 1e-6/token).

All gates run CPU-side against the endpoint; the GPU keeps serving.
