# ==============================================================================
# Foundry Model Image: Qwen3.5-35B-A3B
# ==============================================================================
# Thin layer on top of the base llama.cpp image that adds model-specific
# configuration, GPU profiles, and the entrypoint.
#
# Weights are NOT baked in. They are downloaded on first run or mounted
# from the host at /models.
#
# Usage:
#   docker build -t foundry-qwen3.5-35b-a3b .
#   docker run --gpus all -p 8080:8080 -v ~/.cache/foundry:/models \
#     foundry-qwen3.5-35b-a3b
# ==============================================================================

ARG BASE_IMAGE=foundry-base-llama-cpp:latest
FROM ${BASE_IMAGE}

# Model metadata
ENV FOUNDRY_MODEL_NAME="Qwen3.5-35B-A3B"
ENV FOUNDRY_GGUF_REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
ENV FOUNDRY_GGUF_FILE="Qwen3.5-35B-A3B-Q4_K_M.gguf"

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

EXPOSE 8080

ENTRYPOINT ["/opt/foundry/entrypoint.sh"]
