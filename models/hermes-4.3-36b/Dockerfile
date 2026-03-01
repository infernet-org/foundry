# ==============================================================================
# Foundry Model Image: Hermes-4.3-36B
# ==============================================================================
# Built on top of the official llama.cpp server image (CUDA 12).
# Adds huggingface-hub for model download, GPU auto-tuning profiles, and the
# Foundry entrypoint.
#
# Weights are NOT baked in. They are downloaded on first run or mounted
# from the host at /models.
#
# Usage:
#   docker build -t foundry-hermes-4.3-36b .
#   docker run --gpus all -p 8080:8080 -v ~/.cache/foundry:/models \
#     foundry-hermes-4.3-36b
# ==============================================================================

FROM ghcr.io/ggml-org/llama.cpp:server-cuda12

# Install huggingface-hub for model download (~50MB layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && pip3 install --no-cache-dir "huggingface-hub>=0.28,<1" "hf_transfer>=0.1.6" \
    && rm -rf /var/lib/apt/lists/*

# Model metadata
ENV FOUNDRY_MODEL_NAME="Hermes-4.3-36B"
ENV FOUNDRY_GGUF_REPO="bartowski/NousResearch_Hermes-4.3-36B-GGUF"
ENV FOUNDRY_GGUF_FILE="NousResearch_Hermes-4.3-36B-Q4_K_M.gguf"

# Enable fast downloads
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Runtime defaults (can be overridden)
ENV FOUNDRY_PROFILE="auto"
ENV FOUNDRY_PORT="8080"
ENV FOUNDRY_CTX_LENGTH=""
ENV FOUNDRY_THREADS=""
ENV FOUNDRY_EXTRA_ARGS=""

# Copy profiles and entrypoint
COPY profiles/ /opt/foundry/profiles/
COPY entrypoint.sh /opt/foundry/entrypoint.sh
RUN chmod +x /opt/foundry/entrypoint.sh

# Model storage
RUN mkdir -p /models
VOLUME /models

EXPOSE 8080

ENTRYPOINT ["/opt/foundry/entrypoint.sh"]
