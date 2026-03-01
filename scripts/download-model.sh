#!/bin/bash
# ==============================================================================
# Foundry: Download model GGUF
# ==============================================================================
# Usage:
#   ./scripts/download-model.sh                                    # Default (qwen3.5-35b-a3b)
#   ./scripts/download-model.sh --model hermes-4.3-36b             # Hermes model
#   ./scripts/download-model.sh --model qwen3.5-35b-a3b --quant Q8_0
#   ./scripts/download-model.sh --output /path/to/dir
# ==============================================================================

set -euo pipefail

MODEL="qwen3.5-35b-a3b"
QUANT=""
OUTPUT_DIR="${HOME}/.cache/foundry"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --quant)
            QUANT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --repo)
            REPO_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL] [--quant QUANT] [--output DIR] [--repo REPO]"
            exit 1
            ;;
    esac
done

# --- Model registry ----------------------------------------------------------
# Maps model name -> (repo, filename_pattern)

case "$MODEL" in
    qwen3.5-35b-a3b)
        REPO="${REPO_OVERRIDE:-unsloth/Qwen3.5-35B-A3B-GGUF}"
        QUANT="${QUANT:-UD-Q4_K_XL}"
        FILENAME="Qwen3.5-35B-A3B-${QUANT}.gguf"
        ;;
    hermes-4.3-36b)
        REPO="${REPO_OVERRIDE:-bartowski/NousResearch_Hermes-4.3-36B-GGUF}"
        QUANT="${QUANT:-Q4_K_M}"
        FILENAME="NousResearch_Hermes-4.3-36B-${QUANT}.gguf"
        ;;
    *)
        echo "ERROR: Unknown model '${MODEL}'"
        echo "Available models: qwen3.5-35b-a3b, hermes-4.3-36b"
        exit 1
        ;;
esac

FILEPATH="${OUTPUT_DIR}/${FILENAME}"

mkdir -p "$OUTPUT_DIR"

if [ -f "$FILEPATH" ]; then
    SIZE=$(du -h "$FILEPATH" | cut -f1)
    echo "Model already exists: ${FILEPATH} (${SIZE})"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading ${FILENAME} from ${REPO}..."
echo "Output: ${OUTPUT_DIR}/"
echo ""

if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download \
        "${REPO}" \
        "${FILENAME}" \
        --local-dir "${OUTPUT_DIR}"
else
    echo "huggingface-cli not found. Installing..."
    pip install --quiet huggingface-hub
    huggingface-cli download \
        "${REPO}" \
        "${FILENAME}" \
        --local-dir "${OUTPUT_DIR}"
fi

if [ -f "$FILEPATH" ]; then
    SIZE=$(du -h "$FILEPATH" | cut -f1)
    echo ""
    echo "Download complete: ${FILEPATH} (${SIZE})"
else
    echo "ERROR: Download failed. File not found at ${FILEPATH}"
    exit 1
fi
