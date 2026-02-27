# ==============================================================================
# Foundry Model Image: Qwen3.5-35B-A3B
# ==============================================================================
# Built on top of the official llama.cpp server image (CUDA 13, Blackwell-ready).
# Adds huggingface-cli for model download, GPU auto-tuning profiles, and the
# Foundry entrypoint.
#
# Weights are NOT baked in. They are downloaded on first run or mounted
# from the host at /models.
#
# Usage:
#   docker build -t foundry-qwen3.5-35b-a3b .
#   docker run --gpus all -p 8080:8080 -v ~/.cache/foundry:/models \
#     foundry-qwen3.5-35b-a3b
# ==============================================================================

FROM ghcr.io/ggml-org/llama.cpp:server-cuda12

# Install huggingface-cli for model download (~50MB layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && pip3 install --no-cache-dir huggingface-hub \
    && rm -rf /var/lib/apt/lists/*

# Model metadata
ENV FOUNDRY_MODEL_NAME="Qwen3.5-35B-A3B"
ENV FOUNDRY_GGUF_REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
ENV FOUNDRY_GGUF_FILE="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"

# Runtime defaults (can be overridden)
ENV FOUNDRY_PROFILE="auto"
ENV FOUNDRY_PORT="8080"
ENV FOUNDRY_CTX_LENGTH=""
ENV FOUNDRY_THREADS=""
ENV FOUNDRY_EXTRA_ARGS=""

# Copy profiles and entrypoint
COPY profiles/ /opt/foundry/profiles/
COPY entrypoint.sh /opt/foundry/entrypoint.sh
RUN chmod +x /opt/foundry/entrypoint.sh /opt/foundry/profiles/*.sh

# Model storage
RUN mkdir -p /models
VOLUME /models

EXPOSE 8080

ENTRYPOINT ["/opt/foundry/entrypoint.sh"]
