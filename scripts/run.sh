#!/bin/bash
# ==============================================================================
# Foundry: Run a model container
# ==============================================================================
# Usage:
#   ./scripts/run.sh                          # Auto-detect GPU
#   ./scripts/run.sh --profile rtx4090        # Explicit profile
#   ./scripts/run.sh --port 9090              # Custom port
#   ./scripts/run.sh --detach                 # Run in background
# ==============================================================================

set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io/infernet-org/foundry}"
MODEL="qwen3.5-35b-a3b"
PORT="${PORT:-8080}"
PROFILE="auto"
MODELS_DIR="${MODELS_DIR:-${HOME}/.cache/foundry}"
DETACH=""
EXTRA_DOCKER_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --detach|-d)
            DETACH="-d"
            shift
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$MODELS_DIR"

echo "Starting Foundry (${MODEL})..."
echo "  Profile: ${PROFILE}"
echo "  Port:    ${PORT}"
echo "  Models:  ${MODELS_DIR}"
echo ""

docker run --gpus all \
    --shm-size 2g \
    -p "${PORT}:8080" \
    -v "${MODELS_DIR}:/models" \
    -e FOUNDRY_PROFILE="${PROFILE}" \
    --name "foundry-${MODEL}" \
    --rm \
    ${DETACH} \
    ${EXTRA_DOCKER_ARGS} \
    "${REGISTRY}/${MODEL}:latest"
