---
title: "Reference Code Retrieval"
---

# Reference Code Retrieval

This document describes the design of EdgeQuake’s reference-code retrieval system: how code is indexed, what `query_code` is meant to do, how it ranks results, and how algorithm context changes its behavior.

## Purpose

Paper retrieval answers questions about what a paper says.

Reference-code retrieval answers questions about how the paper’s implementation works.

It exists for agent tasks such as:

- porting a reference implementation
- locating the function behind a paper algorithm
- understanding helper code around an approved algorithm
- tracing symbol relationships after a search hit

It is not meant for prose questions about the paper itself.

## What gets indexed

Reference-code retrieval operates on approved paper repos that have been indexed into a separate code graph.

An index stores:

- symbol-level code chunks
- symbol metadata such as file, line span, and language
- symbol-to-symbol edges such as calls, imports, references, inheritance, and implementation
- anchor information linking approved paper algorithms to specific code artifacts

Two index modes exist:

- `algorithm_focused`: keep code around approved anchors and their containing files
- `full`: index the full repo for exploration

The important design point is that anchor chunks and surrounding chunks are distinct. A file can contain:

- exact algorithm anchor chunks
- many nearby non-anchor symbol chunks

That distinction matters for `query_code`.

## `query_code` behavior

`query_code` is a hybrid code search endpoint surfaced through MCP.

Inputs:

- natural-language query
- optional `document_repo_id`
- optional `index_id`
- optional `algorithm_ids`
- optional `limit`
- optional `max_distance`

Output:

- ranked code chunks with file path, line range, symbol name, language, content, and score signals

The tool is optimized for “find the relevant implementation snippet” rather than “browse the whole graph”.

## Retrieval pipeline

`query_code` combines three signals.

### 1. Entity expansion

The query is parsed for code-shaped identifiers, such as:

- `AttentionHead`
- `rebalance_clusters`
- `` `compute_dot` ``
- `seal::Encryptor::encrypt`

Those identifiers are matched directly against indexed symbol names using:

- exact match
- prefix match for qualified children
- suffix match for qualified parents

This path ensures exact symbol-style queries do not depend on embedding quality.

### 2. Vector retrieval

The natural-language query is embedded with the code embedder and searched against indexed code chunk embeddings.

This is the main recall path for questions like:

- “how is the decrypt procedure initialized”
- “find the distance computation helper”
- “where is the reranking step implemented”

Vector search is bounded by `max_distance` and by the retrieval scope filters.

### 3. BM25 reranking on the candidate set

After entity and vector candidates are merged, the system builds an in-memory BM25 index over that candidate set and reranks with a code-aware tokenizer.

The tokenizer splits:

- camelCase
- PascalCase
- snake_case
- digit boundaries
- punctuation

This improves exact-token behavior without maintaining a persistent full-corpus sparse index.

## Final ranking

The final score blends:

- vector similarity
- BM25 score
- entity-match boost

This gives `query_code` three useful properties:

- semantic queries still work
- exact identifier hits rise to the top
- lexical precision is added without replacing dense retrieval

## How `algorithm_ids` works

This is the key behavior change.

`algorithm_ids` is not an exact-anchor-chunk filter anymore.

It now means:

- resolve the approved anchor chunk(s) for the given algorithms
- resolve their anchor symbol(s)
- walk the indexed symbol graph outward through the full connected neighborhood
- make every reachable symbol-linked chunk eligible for search

That scope is applied consistently to:

- vector retrieval
- entity expansion

So when an agent passes `algorithm_ids` as high-level context, `query_code` can retrieve:

- the algorithm anchor itself
- helper functions
- initialization code
- callers and nearby implementation logic

Example intent:

- the agent knows the SAP decrypt algorithm id
- the agent asks how decrypt is initialized by the SAP scheme
- `query_code` can now return surrounding chunks like constructors, utility functions, and distance helpers, not only the exact `decrypt` anchor chunk

This is the intended design: `algorithm_ids` defines algorithm context, not just algorithm identity.

## Scope and filtering rules

Filters combine by intersection.

- `document_repo_id` narrows to one paper repo
- `index_id` narrows to one concrete index build
- `algorithm_ids` narrows to the graph-connected code neighborhood for those algorithms inside the selected repo/index

If `algorithm_ids` resolves to no indexed anchor symbols, the result is empty. The system does not silently fall back to whole-repo search.

## Tradeoffs

### Why candidate-set BM25 instead of full-corpus sparse search

Full-corpus BM25 would improve global IDF quality, but it adds persistence, rebuild, and staleness costs. Candidate-set BM25 is cheaper and good enough for the intended use case: reranking a small hybrid result set for agent consumption.

### Why graph-neighborhood scoping for `algorithm_ids`

Exact-anchor-only filtering was too narrow for real implementation questions. Agents often know the target algorithm but need code around it:

- setup
- helper math
- shared utilities
- call context

Graph-neighborhood scoping captures that surrounding implementation without widening to the whole repo.

### Why no hop cap

The current behavior uses the full connected neighborhood of the algorithm anchor within the selected index. This favors recall and keeps the API simple. If a repo later proves too broad, hop controls can be added as a tuning feature, but the current design intentionally treats `algorithm_ids` as unrestricted algorithm context.

## Usage guidance

Use `query_code` when you want implementation evidence, not paper prose.

Good queries:

- `how decrypt is initialized by the SAP scheme`
- `find the attention kernel`
- `show where distance comparison happens`

Useful scoping patterns:

- pass `document_repo_id` when you know the paper repo
- pass `index_id` when you want reproducibility against one specific code index
- pass `algorithm_ids` when the agent knows the relevant algorithm and wants surrounding code context

Then use `get_symbol_neighborhood` after a hit when the task becomes navigational:

- what calls this
- what does this depend on
- what symbols are adjacent to this implementation
