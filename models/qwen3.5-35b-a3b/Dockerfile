# ==============================================================================
# Foundry Model Image: Qwen3.5-35B-A3B
# ==============================================================================
# Multi-stage build for a minimal CUDA runtime.
# Compiles llama.cpp from source for sm_89 (Ada) and sm_120a (Blackwell),
# then copies only the binary and required libraries to a clean Ubuntu base.
#
# Weights are NOT baked in. They are downloaded on first run or mounted
# from the host at /models.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 as builder

RUN apt-get update && apt-get install -y git cmake g++ curl
RUN git clone --depth 1 -b b8183 https://github.com/ggml-org/llama.cpp.git /llama.cpp
WORKDIR /llama.cpp

# Compile explicitly for Ada (sm_89) and Blackwell (sm_120a)
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89;120a" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF && \
    cmake --build build --config Release -j$(nproc) --target llama-server

# ------------------------------------------------------------------------------
# Stage 2: Minimal Runtime
# ------------------------------------------------------------------------------
FROM ubuntu:24.04

# Install minimal runtime dependencies (CUDA runtime libs are mounted by nvidia-container-runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    python3 python3-pip curl \
    && pip3 install --break-system-packages --no-cache-dir "huggingface-hub>=0.28,<1" "hf_transfer>=0.1.6" \
    && rm -rf /var/lib/apt/lists/*

# The NVIDIA runtime needs these env vars to mount the CUDA drivers correctly
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Model metadata
ENV FOUNDRY_MODEL_NAME="Qwen3.5-35B-A3B"
ENV FOUNDRY_GGUF_REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
ENV FOUNDRY_GGUF_FILE="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
ENV FOUNDRY_ARCH="moe"

# Enable fast downloads
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Runtime defaults (can be overridden)
ENV FOUNDRY_PROFILE="auto"
ENV FOUNDRY_PORT="8080"
ENV FOUNDRY_CTX_LENGTH=""
ENV FOUNDRY_THREADS=""
ENV FOUNDRY_EXTRA_ARGS=""

# Copy only the compiled binary and its directly required GGML libraries
COPY --from=builder /llama.cpp/build/bin/llama-server /app/llama-server
COPY --from=builder /llama.cpp/build/ggml/src/libggml.so /app/
COPY --from=builder /llama.cpp/build/ggml/src/ggml-cuda/libggml-cuda.so /app/
COPY --from=builder /llama.cpp/build/src/libllama.so /app/
ENV LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

# Copy profiles and shared entrypoint
COPY profiles/ /opt/foundry/profiles/
COPY entrypoint.sh /opt/foundry/entrypoint.sh
RUN chmod +x /opt/foundry/entrypoint.sh

# Model storage
RUN mkdir -p /models
VOLUME /models

EXPOSE 8080

ENTRYPOINT ["/opt/foundry/entrypoint.sh"]
