# Table representation in RAG — converged design & implementation plan

**Date:** 2026-06-12 · **Branch:** `feat/document-references` · **Status:** design converged, implementation in progress

## Problem

For pre-2026-06-03 docs tables weren't embedded at all (fixed by re-embed). After that, the
remaining quality issues, established empirically against the live XorMM doc:

1. **Attribution** — table chunks surfaced with empty `[]:` source. *(FIXED — `extract_document_id`
   now handles `-table-`.)*
2. **Ranking** — the dedicated table chunk is reranked by the workspace **bge-reranker-v2-m3**
   (verified: response `score` == sigmoid(cross-encoder logit); table logit 3.24 vs prose 4.34),
   and the cross-encoder under-ranks the dense GFM grid vs fluent prose.
3. **Redundancy** — a table existed in three places (inlined into a prose chunk, a dedicated table
   chunk, a graph entity), all embedding different text → index bloat + double-count risk.

## Decisions (all confirmed with the user)

| # | Decision | Rationale (research-backed) |
|---|----------|------|
| A | **Embed-text** of the table chunk = caption + column headers (not the full grid) | markup/numbers embed poorly; caption-led is the competitive signal (TabRAG, TARGET, verbalization study) |
| B | **Rerank-text** (kind=table) = caption + headers + **first column** ("both axes"), stored at ingest; reranker uses it instead of full GFM content | cross-encoders under-rank terse/grid content (SciRerankBench); high-signal axes without numeric-cell noise |
| C | **Table graph entity** embeds the same caption-led text (not full GFM) | consistency across layers; same dilution fix |
| D | **Attribution**: `extract_document_id` handles `-table-` | DONE + regression test |
| E | **Link + hydrate + dedup**: prose keeps a lightweight `![tbl_x]` pointer (no inlined GFM); at assembly, hydrate referenced tables for retrieved prose chunks and **dedup by `table_id`** so each table renders exactly once | parent-document / IndexNode pattern; mirrors how figures already work (reference + fetch on demand) |
| — | **Full GFM** kept only as the table chunk's stored/returned **content** (display fidelity) | embed/rerank ≠ display decoupling (LangChain MultiVector) |

Cell-value queries are served by the dedicated table chunk (caption-led embed + headers/first-col
rerank-text); caption queries by the same chunk; prose queries by the prose chunk, which **hydrates**
its referenced table into context when retrieved.

## Implementation plan (dependency order)

1. **`edgequake-pdf` pure helpers** (`backend/table_extract.rs`): `render_table_embed_text(table)`
   (caption + headers) and `render_table_rerank_text(table)` (caption + headers + first column),
   alongside existing `render_table_markdown` (full GFM, unchanged). + unit tests. *(self-contained, no rebuild to validate)*
2. **Stop inlining GFM into the chunker input** (`pdf_processing.rs:931`): keep the `![tbl_x](edgequake-table)`
   pointer in the prose chunk instead of replacing it with the full GFM. Verify the chunker retains
   the marker in the surrounding prose chunk (treat table placeholder as an inline marker, not a hard
   chunk boundary) so assembly can resolve `prose chunk → table_id`.
3. **Table backfill** (`backfill_table_embeddings`, `backfill_table_entities`): embed `render_table_embed_text`;
   store `content` = full GFM; store `rerank_text` = `render_table_rerank_text` in the vector metadata.
   Entity description embed uses caption-led text too.
4. **Chunk `rerank_text` field**: add to the chunk struct + `build_chunk_from_result` (read from metadata);
   `reranking.rs` uses `chunk.rerank_text` when present (kind=table), else `chunk.content`.
5. **Query-assembly hydration + dedup** (`query_execute.rs`): collect `table_id`s from retrieved table
   chunks; for each retrieved prose chunk, resolve referenced `table_id`s and hydrate the GFM (reuse
   `content.rs` table fetch) for any not already present; dedup so each `table_id` renders once.
6. **Sync upstream `enable_override`** (optional, noted divergence): thread the per-request `enable_rerank`
   into `rerank_chunks_with_strategy` to match `raphaelmansuy/edgequake`.
7. **Go-live**: rebuild the `edgequake` container; re-embed tables (chunks-only reprocess of affected docs,
   with permission). Verify against the live XorMM doc: realistic query returns Table 1, attributed, once.

## Verification

- Probe `mcp/probe-*.mjs` against XorMM (`b4ba9edc…`, workspace `…0003`): realistic query
  "XorMM Comparison with existing volume-hiding EMM schemes" returns Table 1 GFM, attributed
  `[Wang et al. …]:`, exactly once, ranked above unrelated prose.
- `cargo test -p edgequake-pdf --lib table_extract::` for the new helpers.
