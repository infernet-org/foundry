#!/bin/bash
# ==============================================================================
# Foundry: Build Docker images
# ==============================================================================
# Usage:
#   ./scripts/build.sh                    # Build base + model (CUDA 12.8)
#   ./scripts/build.sh --cuda 12.4.1      # Build for CUDA 12.4
#   ./scripts/build.sh --base-only        # Build only the base image
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

REGISTRY="${REGISTRY:-ghcr.io/infernet-org/foundry}"
CUDA_VERSION="12.8.0"
CUDA_ARCH="86;89;120"
BASE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --base-only)
            BASE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Adjust arch for older CUDA
if [[ "$CUDA_VERSION" == 12.4.* ]]; then
    CUDA_ARCH="86;89"
fi

echo "Building base image (CUDA ${CUDA_VERSION}, arch: ${CUDA_ARCH})..."
docker build \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg CUDA_ARCH="${CUDA_ARCH}" \
    -t "${REGISTRY}/base-llama-cpp:latest" \
    "${PROJECT_DIR}/base/llama-cpp/"

if [ "$BASE_ONLY" = true ]; then
    echo "Base image built: ${REGISTRY}/base-llama-cpp:latest"
    exit 0
fi

echo "Building model image (qwen3.5-35b-a3b)..."
docker build \
    --build-arg BASE_IMAGE="${REGISTRY}/base-llama-cpp:latest" \
    -t "${REGISTRY}/qwen3.5-35b-a3b:latest" \
    "${PROJECT_DIR}/models/qwen3.5-35b-a3b/"

echo ""
echo "Build complete:"
echo "  Base:  ${REGISTRY}/base-llama-cpp:latest"
echo "  Model: ${REGISTRY}/qwen3.5-35b-a3b:latest"
echo ""
echo "Run with:"
echo "  docker run --gpus all -p 8080:8080 -v ~/.cache/foundry:/models ${REGISTRY}/qwen3.5-35b-a3b:latest"
