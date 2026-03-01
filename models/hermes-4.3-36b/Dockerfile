# ==============================================================================
# Foundry Model Image: Hermes-4.3-36B
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
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y git cmake g++ curl
RUN git clone --depth 1 -b b8183 https://github.com/ggml-org/llama.cpp.git /llama.cpp
WORKDIR /llama.cpp

# Compile explicitly for Ada (sm_89) and Blackwell (sm_120a).
# GGML_BACKEND_DL=ON builds CUDA as a runtime-loaded plugin (dlopen), which
# avoids the libcuda.so.1 transitive link error during Docker builds where
# no real GPU driver is present. This matches the official llama.cpp Dockerfile.
RUN cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89;120a" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    cmake --build build --config Release -j$(nproc)

# ------------------------------------------------------------------------------
# Stage 2: Minimal Runtime
# ------------------------------------------------------------------------------
FROM ubuntu:24.04

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    python3 python3-pip curl \
    && pip3 install --break-system-packages --no-cache-dir "huggingface-hub>=0.28,<1" "hf_transfer>=0.1.6" \
    && rm -rf /var/lib/apt/lists/*

# The NVIDIA runtime needs these env vars to mount the CUDA drivers correctly
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Model metadata
ENV FOUNDRY_MODEL_NAME="Hermes-4.3-36B"
ENV FOUNDRY_GGUF_REPO="NousResearch/Hermes-4.3-36B-GGUF"
ENV FOUNDRY_GGUF_FILE="hermes-4_3_36b-Q4_K_M.gguf"
ENV FOUNDRY_ARCH="dense"

# Enable fast downloads
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Runtime defaults (can be overridden)
ENV FOUNDRY_PROFILE="auto"
ENV FOUNDRY_PORT="8080"
ENV FOUNDRY_CTX_LENGTH=""
ENV FOUNDRY_THREADS=""
ENV FOUNDRY_EXTRA_ARGS=""

# Copy the compiled binary and all shared libraries from the build output.
# With GGML_BACKEND_DL=ON, backends (ggml-cuda, ggml-cpu-*) are .so modules
# loaded at runtime via dlopen. CMake places everything in build/bin/.
COPY --from=builder /llama.cpp/build/bin/ /app/

# Cherry-pick only the CUDA runtime libs that libggml-cuda.so actually needs.
# libcuda.so.1 is provided by the NVIDIA container runtime at launch.
COPY --from=builder /usr/local/cuda/lib64/libcudart.so.12 /app/
COPY --from=builder /usr/local/cuda/lib64/libcublas.so.12 /app/
COPY --from=builder /usr/local/cuda/lib64/libcublasLt.so.12 /app/
ENV LD_LIBRARY_PATH="/app"

# Copy profiles and shared entrypoint
COPY profiles/ /opt/foundry/profiles/
COPY entrypoint.sh /opt/foundry/entrypoint.sh
RUN chmod +x /opt/foundry/entrypoint.sh

# Model storage
RUN mkdir -p /models
VOLUME /models

EXPOSE 8080

ENTRYPOINT ["/opt/foundry/entrypoint.sh"]
