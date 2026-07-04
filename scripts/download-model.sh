#!/bin/bash
# ==============================================================================
# Foundry: Download model weights
# ==============================================================================
# Downloads the nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot (~22 GB, multi-file
# safetensors) into the foundry model cache so container starts are instant.
#
# Usage:
#   ./scripts/download-model.sh                     # -> ~/.cache/foundry
#   ./scripts/download-model.sh --output /path/dir
# ==============================================================================

set -euo pipefail

REPO="nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_DIR="Qwen3.6-35B-A3B-NVFP4"
OUTPUT_DIR="${HOME}/.cache/foundry"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --repo)
            REPO="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--output DIR] [--repo REPO]"
            exit 1
            ;;
    esac
done

TARGET="${OUTPUT_DIR}/${MODEL_DIR}"

if [ -f "${TARGET}/config.json" ]; then
    echo "Model already present: ${TARGET} ($(du -sh "${TARGET}" | cut -f1))"
    echo "Delete it first if you want to re-download."
    exit 0
fi

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "huggingface-hub not found. Install it with:"
    echo "  pip install huggingface-hub hf_transfer"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
echo "Downloading ${REPO} (~22 GB) to ${TARGET}..."
echo ""

HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
FOUNDRY_REPO="${REPO}" \
FOUNDRY_TARGET="${TARGET}" \
python3 -c "
import os
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
snapshot_download(
    repo_id=os.environ['FOUNDRY_REPO'],
    local_dir=os.environ['FOUNDRY_TARGET'],
    token=token,
)
"

if [ ! -f "${TARGET}/config.json" ]; then
    echo "ERROR: Download failed. config.json not found at ${TARGET}"
    exit 1
fi

echo ""
echo "Download complete: ${TARGET} ($(du -sh "${TARGET}" | cut -f1))"
