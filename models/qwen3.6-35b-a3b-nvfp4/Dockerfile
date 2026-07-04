# ==============================================================================
# Foundry Model Image: Qwen3.6-35B-A3B-NVFP4 (vLLM backend)
# ==============================================================================
# Unlike the llama.cpp-based images, this model uses vLLM. The checkpoint is
# NVIDIA ModelOpt NVFP4 (4-bit floating point) -- a safetensors format that
# llama.cpp cannot load. NVFP4 requires Hopper or Blackwell (RTX 50xx) GPUs.
#
# Weights are NOT baked in. They are downloaded on first run or mounted
# from the host at /models (dir: /models/Qwen3.6-35B-A3B-NVFP4, ~22 GB).
# ==============================================================================

# Pinned: every profile flag below was swept against this exact version
FROM vllm/vllm-openai:v0.24.0

# The NVIDIA runtime needs these env vars to mount the CUDA drivers correctly
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Model metadata
ENV FOUNDRY_MODEL_NAME="Qwen3.6-35B-A3B-NVFP4"
ENV FOUNDRY_HF_REPO="nvidia/Qwen3.6-35B-A3B-NVFP4"
ENV FOUNDRY_MODEL_DIR="Qwen3.6-35B-A3B-NVFP4"

# Enable fast downloads
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Avoid fragmentation OOM during startup: weights (~22 GB) + CUDA graphs +
# compile buffers are a tight fit on 32 GB consumer Blackwell.
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Runtime defaults (can be overridden)
ENV FOUNDRY_PROFILE="auto"
ENV FOUNDRY_PORT="8080"
ENV FOUNDRY_CTX_LENGTH=""
ENV FOUNDRY_EXTRA_ARGS=""

# Model storage
RUN mkdir -p /models
VOLUME /models

# Copy profiles and model-specific entrypoint
COPY profiles/ /opt/foundry/profiles/
COPY --chmod=0755 entrypoint.sh /opt/foundry/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/opt/foundry/entrypoint.sh"]
