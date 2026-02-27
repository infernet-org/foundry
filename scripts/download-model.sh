#!/bin/bash
# ==============================================================================
# Foundry: Download model GGUF
# ==============================================================================
# Usage:
#   ./scripts/download-model.sh                         # Default Q4_K_M
#   ./scripts/download-model.sh --quant Q8_0            # Different quant
#   ./scripts/download-model.sh --output /path/to/dir   # Custom directory
# ==============================================================================

set -euo pipefail

REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
QUANT="Q4_K_M"
OUTPUT_DIR="${HOME}/.cache/foundry"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quant)
            QUANT="$2"
            shift 2
            ;;
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
            exit 1
            ;;
    esac
done

FILENAME="Qwen3.5-35B-A3B-${QUANT}.gguf"
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
        --local-dir "${OUTPUT_DIR}" \
        --local-dir-use-symlinks False
else
    echo "huggingface-cli not found. Installing..."
    pip install --quiet huggingface-hub
    huggingface-cli download \
        "${REPO}" \
        "${FILENAME}" \
        --local-dir "${OUTPUT_DIR}" \
        --local-dir-use-symlinks False
fi

if [ -f "$FILEPATH" ]; then
    SIZE=$(du -h "$FILEPATH" | cut -f1)
    echo ""
    echo "Download complete: ${FILEPATH} (${SIZE})"
else
    echo "ERROR: Download failed. File not found at ${FILEPATH}"
    exit 1
fi
