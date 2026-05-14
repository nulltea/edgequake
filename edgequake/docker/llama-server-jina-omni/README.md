# Jina v5 omni small retrieval — multimodal embedding server

Serves `jinaai/jina-embeddings-v5-omni-small-retrieval` (1024-dim, multimodal:
text + image in a shared vector space, MRL-truncatable). Used by EdgeQuake to
embed PDF text chunks and PDF figure crops in a single pgvector index.

## Why a custom Dockerfile

Stock `ghcr.io/ggml-org/llama.cpp:server-rocm` won't work end-to-end for this
model. Two independent reasons:

1. **Upstream `--embedding --mmproj` is broken for vision GGUFs.** Image bytes
   don't actually flow through `mtmd_encode_chunk` in the embedding path
   (llama.cpp issue #13666, discussion #14851, PR #15108 incomplete).
2. **v5 omni needs Jina-specific patches** on top: `encoder combined-decode`
   (the only path to deliver image features into an encoder-only embedding
   model), bicubic-pillow resize (without it `img_cat` cosine drops from
   ~0.999 to ~0.96 vs torch reference), qwen3vl pos_embed BILINEAR alignment,
   qwen2.5 vision-encoder compat. Index in
   <https://github.com/jina-ai/llama.cpp/blob/feat-v5-omni/JINA-V5-OMNI.md>.

So we build from the Jina fork at `feat-v5-omni`.

## Download GGUFs (~1.3 GB)

```bash
hf download jinaai/jina-embeddings-v5-omni-small-retrieval-GGUF \
    jina-embeddings-v5-omni-small-retrieval-Q8_0.gguf \
    jina-embeddings-v5-omni-small-retrieval-vision-mmproj-F16.gguf \
    --local-dir ~/models/jina-v5-omni-small-retrieval
```

(Q8_0 main quant ~639 MB; F16 vision projector ~660 MB. Skip the audio mmproj
unless you also want to embed audio.)

## Build

Multi-stage Dockerfile:

- **Builder**: `rocm/dev-ubuntu-22.04:7.2.3` — has the HIP compiler
  (`/opt/rocm/llvm/bin/clang++`); we apt install `hipblas-dev`/`rocblas-dev`
  for the cmake step. Clones the Jina `feat-v5-omni` branch and builds only
  the `llama-server` target.
- **Runtime**: `kyuz0/amd-strix-halo-toolboxes:rocm-7.2.3` — the toolbox
  image already running Qwen3 MoE on the host, with ROCm 7.2.3 runtime libs
  validated for gfx1151. We copy the freshly-built `llama-server` over the
  toolbox's upstream one; everything else from the base stays put.

```bash
# Default — Strix Halo iGPU, gfx1151
docker build -t edgequake/jina-omni-llama:rocm docker/llama-server-jina-omni/

# Different AMDGPU target
docker build --build-arg AMDGPU_TARGETS=gfx1100 \
    -t edgequake/jina-omni-llama:rocm-gfx1100 docker/llama-server-jina-omni/
```

The first build is the heavy one (~15–25 min on Strix Halo: apt install
hipblas-dev, clone llama.cpp, compile llama-server with HIP for one gfx
target). Layer cache makes subsequent rebuilds fast as long as the fork
SHA hasn't moved.

## Run

```bash
# GGUFs already downloaded via `hf download` land under:
#   ~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v5-omni-small-retrieval-GGUF/snapshots/<snapshot>/
# Mount that snapshot dir at /models. The Dockerfile defaults to
#   /models/jina-embeddings-v5-omni-small-retrieval-Q8_0.gguf
#   /models/jina-embeddings-v5-omni-small-retrieval-vision-mmproj-F16.gguf

SNAP_DIR=~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v5-omni-small-retrieval-GGUF/snapshots/6bcd03c40b56717ec02ca4a26e9746da505b9ccf

docker run --rm --name jina-omni-llama \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    -v "$SNAP_DIR":/models:ro \
    -p 8082:8082 \
    edgequake/jina-omni-llama:rocm
```

## Smoke tests

Text-only (should return one 1024-dim vector):

```bash
curl -s http://localhost:8082/v1/embeddings -H 'content-type: application/json' -d '{
  "model": "jina-embeddings-v5-omni-small-retrieval",
  "input": [{"type": "text", "text": "Document: hello world"}]
}' | jq '.data[0].embedding | length'
```

Multimodal (caption + image fused; verify the image bytes actually changed the
output vs the text-only call):

```bash
B64=$(base64 -w0 some_figure.png)
curl -s http://localhost:8082/v1/embeddings -H 'content-type: application/json' -d "{
  \"model\": \"jina-embeddings-v5-omni-small-retrieval\",
  \"input\": [[
    {\"type\": \"text\", \"text\": \"Document: figure caption here\"},
    {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$B64\"}}
  ]]
}" | jq '.data[0].embedding | length'
```

Cosine of the fused vector against the text-only vector should be < 0.99 — if
they're identical, the image isn't reaching the encoder (re-check the fork
commit pinned in `/app/JINA_LLAMA_CPP_COMMIT.txt`).

## Wire into EdgeQuake

`models.toml` already has a `jina-omni-local` provider stanza with
`api_base = "http://localhost:8082/v1"`. Set its `enabled = true` and update
your workspace's `embedding_provider` / `embedding_model`. A workspace
configured for a different embedding dimension (768 / 1536) will get its
vector table dropped+recreated on first query — there is no in-place migration.
