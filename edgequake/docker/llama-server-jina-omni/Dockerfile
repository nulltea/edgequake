# Jina v5 omni small retrieval — multimodal embedding server.
#
# Multi-stage build:
#   - Builder:  `rocm/dev-ubuntu-22.04:7.2.3` (has clang++ + hip-dev; we apt
#               install hipblas-dev/rocblas-dev for the cmake config step).
#   - Runtime:  `kyuz0/amd-strix-halo-toolboxes:rocm-7.2.3` — the same
#               toolbox image already running Qwen3 MoE on the host, with
#               ROCm 7.2.3 runtime libs validated for gfx1151.
#
# Why not single-stage on the kyuz0 image: it ships runtime libs but no
# clang/cmake/git, so cmake can't find HIPCXX.
#
# Why not stock llama-server (ghcr.io/ggml-org/llama.cpp:server-rocm):
# upstream `--embedding --mmproj` silently drops image features for
# encoder-only embedding models (llama.cpp issues #13666, #14851, PR #15108
# incomplete) and v5 omni needs Jina's qwen3vl numerical-parity patches —
# `encoder combined-decode`, bicubic-pillow resize (without it `img_cat` cos
# drops from ~0.999 to ~0.96 vs torch), qwen3vl pos_embed BILINEAR, qwen2.5
# vision compat, image_resize_pad portrait fix. See
# https://github.com/jina-ai/llama.cpp/blob/feat-v5-omni/JINA-V5-OMNI.md.
#
# GGUFs are NOT baked in. Mount the directory holding both quants when
# running:
#   ~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v5-omni-small-retrieval-GGUF/snapshots/<snapshot>/

# ─── Builder ─────────────────────────────────────────────────────────────────
FROM rocm/dev-ubuntu-22.04:7.2.3 AS builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build pkg-config ca-certificates \
        libcurl4-openssl-dev \
        hipblas-dev rocblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 50 --branch feat-v5-omni https://github.com/jina-ai/llama.cpp.git . \
    && git log -1 --pretty='%H %s' > /src/COMMIT.txt

ENV HIPCXX=/opt/rocm/llvm/bin/clang++
ENV CMAKE_PREFIX_PATH=/opt/rocm
# gfx1151 = Strix Halo's Radeon 8060S. Override at build time for other GPUs.
ARG AMDGPU_TARGETS=gfx1151
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP=ON \
        -DGGML_HIPBLAS=ON \
        -DGGML_CUDA_FORCE_MMQ=OFF \
        -DAMDGPU_TARGETS=${AMDGPU_TARGETS} \
        -DLLAMA_CURL=ON \
    && cmake --build build --target llama-server -j$(nproc)

# ─── Runtime ─────────────────────────────────────────────────────────────────
# kyuz0 toolbox has the ROCm 7.2.3 runtime libs (hipblas, rocblas, hipblaslt,
# amd_comgr, etc.) already linked against the same gfx1151 stack the host
# uses for other inference workloads. We replace the toolbox's upstream
# llama-server with the Jina-fork build; everything else from the base image
# stays put.
FROM kyuz0/amd-strix-halo-toolboxes:rocm-7.2.3 AS runtime

COPY --from=builder /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /src/COMMIT.txt /opt/JINA_LLAMA_CPP_COMMIT.txt
# Side libs that ggml-hip may have built next to llama-server — copy
# defensively so runtime dlopen finds them. Already-present libs in
# /opt/rocm/lib win.
COPY --from=builder /src/build/bin/ /opt/jina-llama-extras/

# Defaults — override at run time with -e.
ENV LLAMA_HOST=0.0.0.0
ENV LLAMA_PORT=8082
ENV LLAMA_POOLING=last
ENV LLAMA_NGL=999
# Filenames inside the mounted /models dir.
ENV LLAMA_MODEL=/models/jina-embeddings-v5-omni-small-retrieval-Q8_0.gguf
ENV LLAMA_MMPROJ=/models/jina-embeddings-v5-omni-small-retrieval-vision-mmproj-F16.gguf
ENV LD_LIBRARY_PATH=/opt/jina-llama-extras:/opt/rocm/lib

EXPOSE 8082

# `--embedding` puts the server in pooling/embedding mode; `--pooling last`
# matches the v5 omni torch reference. `--mmproj` loads the vision encoder
# so image inputs reach the encoder via Jina's combined-decode patch.
ENTRYPOINT ["/bin/sh", "-c", "exec /usr/local/bin/llama-server \
    -m \"$LLAMA_MODEL\" \
    --mmproj \"$LLAMA_MMPROJ\" \
    --embedding \
    --pooling \"$LLAMA_POOLING\" \
    --host \"$LLAMA_HOST\" \
    --port \"$LLAMA_PORT\" \
    -ngl \"$LLAMA_NGL\""]
