# ==============================================================================
# Foundry Base Image: llama.cpp with CUDA support
# ==============================================================================
# Multi-stage build that compiles llama.cpp from source and produces a slim
# runtime image with just the llama-server binary.
#
# Build args:
#   CUDA_VERSION  - CUDA toolkit version (default: 12.8.0)
#   CUDA_ARCH     - Target GPU architectures (default: 86;89;120)
#   LLAMA_CPP_REF - llama.cpp git ref to build (default: master)
#
# Usage:
#   docker build -t foundry-base-llama-cpp:cu128 .
#   docker build --build-arg CUDA_VERSION=12.4.1 --build-arg CUDA_ARCH="86;89" \
#     -t foundry-base-llama-cpp:cu124 .
# ==============================================================================

ARG CUDA_VERSION=12.8.0

# -----------------------------------------------------------------------------
# Stage 1: Build llama.cpp from source
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build

ARG CUDA_ARCH="86;89;120"
ARG LLAMA_CPP_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
    https://github.com/ggerganov/llama.cpp.git

WORKDIR /build/llama.cpp

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc) --target llama-server

# -----------------------------------------------------------------------------
# Stage 2: Slim runtime image
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    && pip3 install --no-cache-dir huggingface-hub \
    && rm -rf /var/lib/apt/lists/*

# Copy llama-server binary and required libraries
COPY --from=build /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /build/llama.cpp/build/src/libllama.so /usr/local/lib/
COPY --from=build /build/llama.cpp/build/ggml/src/libggml*.so /usr/local/lib/

# Ensure shared libs are found
RUN ldconfig

# Healthcheck
COPY healthcheck.sh /opt/foundry/healthcheck.sh
RUN chmod +x /opt/foundry/healthcheck.sh
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD /opt/foundry/healthcheck.sh

# Default model storage
RUN mkdir -p /models
VOLUME /models

EXPOSE 8080

LABEL org.opencontainers.image.source="https://github.com/infernet-org/foundry"
LABEL org.opencontainers.image.description="Foundry base image: llama.cpp with CUDA"
LABEL org.opencontainers.image.licenses="Apache-2.0"
