#!/usr/bin/env python3
"""FastAPI sidecar exposing OpenAI-style /v1/rerank for jina-reranker-v3.

Loads the model in-process via llama-cpp-python (no subprocess, no stdout
parsing). Each /v1/rerank call:
  1. Formats query + N docs into a single joint prompt with the model's
     special tokens (<|rerank_token|> for the query, <|embed_token|> per doc).
  2. Runs ONE forward pass with pooling=NONE, returning per-token hidden
     states.
  3. Slices the hidden states at the special-token positions and applies the
     2-layer MLP projector (loaded from safetensors) + cosine similarity.

n_ubatch is pinned to n_ctx so the entire prompt is processed in a single
ubatch — this is the safe configuration for NONE pooling and side-steps a
known mean-pooling+ubatch interaction in upstream llama.cpp.

CLI: accepts the same flags as llama-server (-m, -c, -ngl, --port, etc.) so
the llama-swap config entry looks structurally identical to other llama.cpp
models and the dashboard's regex-based -c/-np parser works against it. If no
CLI flags are passed, falls back to MODEL_PATH/PROJECTOR_PATH/CTX_SIZE/NGL
env vars (legacy launch path).
"""
import argparse
import os
import sys
import time
from typing import Dict, List, Optional


def _parse_cli(argv):
    p = argparse.ArgumentParser(
        prog="jina-reranker-v3",
        description="jina-reranker-v3 OpenAI-compatible /v1/rerank server",
    )
    p.add_argument("-m", "--model", required=True, help="GGUF model path")
    p.add_argument(
        "--mmproj", "--projector", dest="mmproj", required=True,
        help="MLP projector safetensors path "
             "(named --mmproj for symmetry with llama.cpp multimodal projectors)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=11443)
    p.add_argument(
        "-c", "--ctx-size", dest="ctx_size", type=int, default=65536,
        help="context size — pinned to a single sequence (n_seq_max=1)",
    )
    p.add_argument(
        "-ngl", "--n-gpu-layers", dest="ngl", type=int, default=99,
        help="layers offloaded to GPU",
    )
    p.add_argument("-a", "--alias", default="jina-reranker-v3")
    p.add_argument("--threads", type=int, default=0)
    # llama-server-style flags we accept silently so the cmd-line stays
    # structurally uniform across llama-swap entries:
    p.add_argument("--ubatch-size", type=int, default=2048,
                   help="physical prefill chunk size — sized for compute "
                        "scratch, not the full prompt. 2048 is plenty for "
                        "throughput; smaller saves VRAM. n_batch always = n_ctx.")
    p.add_argument("-np", "--parallel", type=int, default=1,
                   help="ignored — n_seq_max is forced to 1")
    p.add_argument("--no-mmap", action="store_true",
                   help="ignored — llama-cpp-python decides")
    p.add_argument("--embeddings", action="store_true",
                   help="ignored — always on for this server")
    p.add_argument("--pooling", default=None,
                   help="ignored — always NONE for per-token late-interaction")
    return p.parse_args(argv)


# Only run the CLI parser when the script is invoked with a model arg. When
# the module is re-imported (e.g. by `uvicorn --app-dir /app sidecar:app`),
# fall back to env vars so the legacy launch path keeps working.
_CLI_FLAGS = ("-m", "--model")
if any(flag in sys.argv for flag in _CLI_FLAGS):
    _args = _parse_cli(sys.argv[1:])
    os.environ["MODEL_PATH"] = _args.model
    os.environ["PROJECTOR_PATH"] = _args.mmproj
    os.environ["CTX_SIZE"] = str(_args.ctx_size)
    os.environ["UBATCH_SIZE"] = str(_args.ubatch_size)
    os.environ["NGL"] = str(_args.ngl)
    if _args.threads:
        os.environ["N_THREADS"] = str(_args.threads)
    os.environ["HOST"] = _args.host
    os.environ["PORT"] = str(_args.port)
else:
    _args = None


import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from safetensors import safe_open
import llama_cpp
from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE


MODEL_PATH = os.environ["MODEL_PATH"]
PROJECTOR_PATH = os.environ["PROJECTOR_PATH"]
NGL = int(os.environ.get("NGL", "99"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "65536"))
UBATCH_SIZE = int(os.environ.get("UBATCH_SIZE", "2048"))
N_THREADS = int(os.environ.get("N_THREADS", "0")) or None

# Token IDs from the jina-reranker-v3 tokenizer (added_tokens.json).
QUERY_EMBED_TOKEN = "<|rerank_token|>"
DOC_EMBED_TOKEN = "<|embed_token|>"
QUERY_EMBED_TOKEN_ID = 151671
DOC_EMBED_TOKEN_ID = 151670


def _load_projector(path: str):
    with safe_open(path, framework="numpy") as f:
        w0 = f.get_tensor("projector.0.weight")  # [hidden, intermediate]
        w2 = f.get_tensor("projector.2.weight")  # [intermediate, out]
    return w0, w2


def _project(x: np.ndarray, w0: np.ndarray, w2: np.ndarray) -> np.ndarray:
    x = x @ w0.T
    x = np.maximum(0, x)
    x = x @ w2.T
    return x


def _sanitize(text: str) -> str:
    return text.replace(QUERY_EMBED_TOKEN, "").replace(DOC_EMBED_TOKEN, "")


def _format_prompt(query: str, docs: List[str], instruction: Optional[str]) -> str:
    query = _sanitize(query)
    docs = [_sanitize(d) for d in docs]

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"

    body = (
        f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    if instruction:
        body += f"<instruct>\n{instruction}\n</instruct>\n"

    body += "\n".join(
        f'<passage id="{i}">\n{doc}{DOC_EMBED_TOKEN}\n</passage>'
        for i, doc in enumerate(docs)
    ) + "\n"
    body += f"<query>\n{query}{QUERY_EMBED_TOKEN}\n</query>"
    return prefix + body + suffix


W0, W2 = _load_projector(PROJECTOR_PATH)


# llama-cpp-python's Llama() with embedding=True forcibly sets
# context_params.n_seq_max = min(n_batch, llama_max_parallel_sequences())
# at llama.py:401 — capping it at 256. That assumption fits an embedding
# *batch server* (one seq per input). For our case (one joint prompt per
# call), we want all of n_ctx in a single sequence. Patch
# llama_max_parallel_sequences to return 1 so the min() lands on 1.
# Patch on the submodule llama_cpp.llama_cpp (the ctypes bindings), since
# llama_cpp/llama.py imports it as `import llama_cpp.llama_cpp as llama_cpp`
# — so the name resolution inside Llama() goes through the submodule.
import llama_cpp.llama_cpp as _llc
_orig_max_parallel = _llc.llama_max_parallel_sequences
_llc.llama_max_parallel_sequences = lambda: 1
try:
    # n_batch stays large so a single .embed() call can ingest a 30K+ token
    # joint prompt without splitting at the API layer. n_ubatch is the
    # *physical* prefill chunk that hits the GPU per step — keeping it small
    # cuts compute-buffer VRAM dramatically (the buffer is sized for ubatch,
    # not the full prompt). NONE pooling doesn't care about ubatch size.
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=CTX_SIZE,
        n_batch=CTX_SIZE,
        n_ubatch=UBATCH_SIZE,
        n_gpu_layers=NGL,
        n_threads=N_THREADS,
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_NONE,
        logits_all=True,
        flash_attn=True,
        verbose=True,
    )
finally:
    _llc.llama_max_parallel_sequences = _orig_max_parallel

app = FastAPI()


class RerankRequest(BaseModel):
    model: Optional[str] = None  # accepted for OpenAI/Jina compatibility, ignored
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    instruction: Optional[str] = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.documents:
        return RerankResponse(results=[])

    prompt = _format_prompt(req.query, req.documents, req.instruction)

    # We tokenize ourselves with special=True (so <|rerank_token|> /
    # <|embed_token|> stay as single IDs), then drive the low-level batch +
    # decode directly. llm.embed() would re-tokenize with special=False
    # internally and split our control tokens into sub-word pieces, throwing
    # off the position mapping.
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    n_tokens = len(tokens)
    n_embd = llm.n_embd()

    if n_tokens > CTX_SIZE:
        raise HTTPException(
            500,
            detail=f"prompt too long: {n_tokens} tokens > ctx_size {CTX_SIZE}",
        )

    try:
        llm._batch.reset()
        llm._batch.add_sequence(tokens, 0, True)  # seq_id=0, logits_all=True
        llm._ctx.kv_cache_clear()
        _t0 = time.perf_counter()
        llm._ctx.decode(llm._batch)
        _dt = time.perf_counter() - _t0
        print(
            f"[perf] prefill: {n_tokens} tokens in {_dt:.2f}s = "
            f"{n_tokens / _dt:.1f} tok/s",
            flush=True,
        )
    except Exception as e:
        raise HTTPException(
            500,
            detail=f"forward pass failed: {type(e).__name__}: {e}",
        )

    ptr = _llc.llama_get_embeddings(llm._ctx.ctx)
    flat = np.array(ptr[: n_tokens * n_embd], dtype=np.float32)
    hidden = flat.reshape(n_tokens, n_embd)
    tokens = np.asarray(tokens, dtype=np.int64)

    query_positions = np.where(tokens == QUERY_EMBED_TOKEN_ID)[0]
    doc_positions = np.where(tokens == DOC_EMBED_TOKEN_ID)[0]

    if query_positions.size == 0:
        raise HTTPException(500, "query embed token not found in tokenized prompt")
    if doc_positions.size != len(req.documents):
        raise HTTPException(
            500,
            f"doc embed token count mismatch: got {doc_positions.size}, expected {len(req.documents)}",
        )

    query_h = hidden[query_positions[0]:query_positions[0] + 1]
    doc_h = hidden[doc_positions]

    q = _project(query_h, W0, W2)               # [1, out]
    d = _project(doc_h, W0, W2)                 # [N, out]

    dot = (d * q).sum(axis=-1)
    qn = np.linalg.norm(q, axis=-1)
    dn = np.linalg.norm(d, axis=-1)
    scores = dot / (dn * qn + 1e-12)

    ranked = sorted(
        ((i, float(s)) for i, s in enumerate(scores)),
        key=lambda x: x[1],
        reverse=True,
    )
    if req.top_n is not None:
        ranked = ranked[: req.top_n]

    return RerankResponse(
        results=[RerankResult(index=i, relevance_score=s) for i, s in ranked]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "11443")),
        log_level="info",
    )
