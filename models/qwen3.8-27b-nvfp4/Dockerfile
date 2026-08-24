# ==============================================================================
# Foundry Model Image: Qwen3.8-27B-NVFP4 (vLLM backend)
# ==============================================================================
# Dense hybrid-attention member of the Qwen3.8 family (Qwen3_5 architecture:
# 16/64 full-attention layers + 48 linear-attention layers, vision tower,
# built-in MTP draft head, 262K native context). Served from the Unsloth
# dynamic NVFP4 checkpoint (mixed FP8 channel-wise + 4-bit groups).
#
# NVFP4 requires Blackwell (RTX 50xx, sm_120) or Hopper (sm_90).
# NOTE: unlike Qwen3.6, this checkpoint is vLLM auto-detected -- do NOT force
# --quantization modelopt (the entrypoint only passes it if
# FOUNDRY_QUANTIZATION is set).
#
# Weights are NOT baked in. They are downloaded on first run or mounted
# from the host at /models (dir: /models/Qwen3.8-27B-NVFP4, ~22 GB).
# ==============================================================================

# Pinned: profile flags are swept against this exact version.
# >=0.25 required for the Qwen3.8 family; 0.27.x verified against the
# published vLLM recipe for Qwen3.8-27B on consumer Blackwell.
FROM vllm/vllm-openai:v0.27.1

# The checkpoint config.json was written by transformers 5.8.0; matching it
# provides the Qwen3-VL processor classes vLLM needs to parse multimodal
# config (vLLM parses model weights with its own Qwen3_5Config).
RUN pip install --no-cache-dir "transformers>=5.8.0"

# The NVIDIA runtime needs these env vars to mount the CUDA drivers correctly
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Model metadata
ENV FOUNDRY_MODEL_NAME="Qwen3.8-27B-NVFP4"
ENV FOUNDRY_HF_REPO="unsloth/Qwen3.8-27B-NVFP4"
ENV FOUNDRY_MODEL_DIR="Qwen3.8-27B-NVFP4"

# Enable fast downloads
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Reduce fragmentation during startup: weights (~22 GB) + KV pool + eager
# buffers are a tight fit on 32 GB consumer Blackwell.
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Runtime defaults (can be overridden)
ENV FOUNDRY_PROFILE="auto"
ENV FOUNDRY_PORT="8080"
ENV FOUNDRY_CTX_LENGTH=""
ENV FOUNDRY_EXTRA_ARGS=""
ENV FOUNDRY_QUANTIZATION=""

# Model storage
RUN mkdir -p /models
VOLUME /models

# Copy profiles and model-specific entrypoint
COPY profiles/ /opt/foundry/profiles/
COPY --chmod=0755 entrypoint.sh /opt/foundry/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/opt/foundry/entrypoint.sh"]
