# jina-reranker-v3 rerank server (llama-cpp-python + FastAPI sidecar).
#
# The previous design shelled out to llama-embedding (CLI binary) and parsed
# its stdout JSON. That breaks at >~1M LOG() calls because llama.cpp's
# embedding writer uses the async log queue to emit per-token output, and
# the queue races under high pressure (see jina-reranker-v3 debug notes).
#
# Now: load the model in-process via llama-cpp-python (Python binding to
# libllama). Per-token hidden states come back as a numpy array directly,
# no subprocess, no stdout parsing, no async-log-queue issues. The model
# stays warm across requests.
#
# Why upstream llama-cpp-python instead of Hanxiao's fork:
#   - We apply the safetensors MLP projector + cosine ourselves in Python,
#     so we don't need Hanxiao's projector-baking or RANK pooling.
#   - With pooling_type=NONE we get per-token rows directly.
#   - Hanxiao's only substantive patch (mean-pooling + ubatch bug) doesn't
#     affect NONE pooling provided n_ubatch >= n_ctx.

FROM rocm/dev-ubuntu-22.04:7.2.3 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Build deps for llama-cpp-python (scikit-build-core compiles llama.cpp from
# its bundled submodule on install). Python 3.11 from deadsnakes — Ubuntu
# 22.04 ships 3.10 by default which is fine; we'll just use the system one.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git build-essential cmake ninja-build pkg-config ca-certificates \
        libcurl4-openssl-dev \
        hipblas-dev rocblas-dev rocwmma-dev \
    && rm -rf /var/lib/apt/lists/*

ENV HIPCXX=/opt/rocm/llvm/bin/clang++
ENV CMAKE_PREFIX_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:$PATH

# Strix Halo gfx1151 build flags (refs: llama.cpp discussions #20856, #22411
# and issues #21284, #13565; lemonade-sdk/llamacpp-rocm #7):
#   NO_VMM           — gfx1151's HIP VMM allocator is broken; required
#   ROCWMMA_FATTN    — matrix-core accelerated flash-attention; biggest
#                      prefill uplift on this hardware
#   MMQ_MFMA         — MFMA path for quantized matmul (Q4/Q8 weights);
#                      reportedly +40-60% pp on gfx1151 vs default
ARG AMDGPU_TARGETS=gfx1151
ENV CMAKE_ARGS="-DGGML_HIP=on -DGGML_HIP_NO_VMM=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_HIP_MMQ_MFMA=ON -DAMDGPU_TARGETS=${AMDGPU_TARGETS} -DGPU_TARGETS=${AMDGPU_TARGETS}"
ENV FORCE_CMAKE=1

# Install llama-cpp-python from source (no prebuilt wheel for ROCm). This
# compiles llama.cpp's bundled submodule against ROCm/HIP — expect 5-15
# minutes on the iGPU box. Pin to a known-good version.
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir --no-binary llama-cpp-python \
        "llama-cpp-python==0.3.23"

# Sidecar deps. Keep numpy on whatever pip resolves alongside llama-cpp-python.
RUN pip3 install --no-cache-dir \
        "fastapi>=0.111" "uvicorn[standard]>=0.30" \
        "pydantic>=2.6" "safetensors>=0.4"

WORKDIR /app
COPY sidecar.py /app/sidecar.py
RUN chmod +x /app/sidecar.py

# Defaults consumed only when the entrypoint is invoked WITHOUT CLI flags
# (legacy launch). The llama-swap entry passes -m / --mmproj / -c / -ngl /
# --port directly, which override these.
ENV PORT=11443
ENV NGL=99
ENV CTX_SIZE=65536
ENV LD_LIBRARY_PATH=/opt/rocm/lib

EXPOSE 11443

# sidecar.py is the canonical entrypoint and accepts llama-server-style
# CLI flags. The llama-swap config invokes it as:
#   docker run ... edgequake/jina-reranker-v3:rocm \
#       -m <gguf> --mmproj <projector.safetensors> \
#       --host 0.0.0.0 --port 11443 -c 65536 -ngl 99 -np 1 ...
ENTRYPOINT ["python3", "/app/sidecar.py"]
