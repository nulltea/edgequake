# EdgeQuake TODO

## Critical: Make auto-recovery mechanism safer

**Priority:** High  
**Date:** 2026-04-17  
**Context:** Server restart auto-recovery overwrote document metadata for documents in algorithm extraction stages (`algo_identifying`, `algo_extracting`, `algo_verifying`, `algo_embedding`), causing loss of `workspace_id`, `title`, and other fields. This made 11 documents disappear from the UI.

### Root cause

1. Algorithm extraction changes a `completed` document's status to intermediate stages (e.g., `algo_identifying`)
2. On server restart, the auto-recovery mechanism sees these as "stuck" documents
3. Recovery overwrites the entire metadata JSON with a minimal placeholder (just `id`, `status`, `stage_message`) — losing `workspace_id`, `title`, `tenant_id`, `source_type`, `pdf_id`, entity/chunk counts, etc.

### Required fixes

- [ ] Auto-recovery must **preserve all existing metadata fields** when resetting status — only update `status`, `current_stage`, `stage_message`, and `updated_at`
- [ ] Auto-recovery should **not touch documents in `algo_*` stages** — these are algorithm extraction stages, not document processing stages. The document itself is already `completed`; the algo extraction is a separate concern
- [ ] Consider storing algorithm extraction status separately from document status (e.g., in a dedicated field or the algorithms table) so document metadata is never modified during algorithm operations
- [ ] Add a safeguard: if a document has `chunk_count > 0` or entities in the graph, never reset it to `pending` — it's clearly been processed

### Files involved

- `edgequake/crates/edgequake-api/src/processor/status_updates.rs` — `update_document_status()` creates new metadata without workspace_id if existing metadata is missing
- Server startup auto-recovery code (grep for "Auto-recovered after server restart")
- `edgequake/crates/edgequake-api/src/processor/algorithm_extraction.rs` — calls `update_document_status()` which modifies document metadata during algo extraction

## Switch to context aware chunker


## Entity-name canonicalization between text and vision extraction

**Priority:** Medium
**Date:** 2026-05-14
**Context:** The figure-entity backfill (`backfill_figure_entities` in `pdf_processing.rs`) extracts entities from figures via the same prompt the text body uses, then merges figure chunk IDs into existing entity-vector rows by exact entity-name match. Most figure-extracted names don't match their text-pass counterparts because the LLM produces different canonical forms across passes.

### Observed mismatches on doc `4f7ce548-…` (ObfuscaTune)

| Text-pass produced | Vision-pass produced | Merge result |
|---|---|---|
| `entity:ObfuscaTune` + `entity:OBFUSCATUNE` | `OBFUSCATUNE` | Only the all-caps variant got figure linkage |
| `entity:GPT-2` + `entity:GPT2` | `GPT-2` | Only the hyphenated variant got figure linkage |
| `entity:Figure 2` | (vision pass didn't emit "Figure 2") | No figure linkage on what would have been a perfect query anchor |
| `entity:GPT2-Large`, `entity:GPT2-Medium` | — | No vision counterpart |

Net effect: query "Detailed architecture of the GPT-2 with M layers using ObfuscaTune" ranks `entity:Figure 2` #1 and `entity:GPT2` #5 by ANN, neither of which has the figure in `source_chunk_ids`. The figure-linked entities (`MLP`, `TEE`, `GPT-2`, `SOFTMAX`) rank #13 and below — outside the entity-driven retrieval's top-K.

### Required fixes

- [ ] Run the text-extraction normalizer (`normalize_entity_name` in `crates/edgequake-pipeline/src/prompts/`) on vision-pass outputs BEFORE the upsert, so both passes converge on the same canonical key
- [ ] When the upsert collapses a duplicate (e.g. both `ObfuscaTune` and `OBFUSCATUNE` exist), merge their `source_chunk_ids` instead of leaving two rows pointing at different chunks
- [ ] In `backfill_figure_entities`'s `merge_figure_chunk_ids_into_entity_vectors`, also try fuzzy candidates: case-insensitive match on `metadata->>'entity_name'`, with-and-without-hyphen variants — so figure passes can patch onto text-extracted entities even when names diverge by punctuation
- [ ] As a heuristic-only fallback: when the figure caption text contains `Figure N` / `Fig. N` / `Table N`, look up any `entity:Figure N` already in the vector store and append the figure chunk_id to its `source_chunk_ids`. Caption-anchor entities are commonly query anchors

### Files involved

- `crates/edgequake-pipeline/src/prompts/` — entity name normalization
- `crates/edgequake-api/src/processor/pdf_processing.rs::backfill_figure_entities` — merge step
- `crates/edgequake-api/src/processor/text_insert.rs:756-806` — text-pass upsert (for the deduplication step)

> NOTE: This issue affects hybrid retrieval recall for figure chunks specifically. The architectural fix (add direct chunk-ANN as a third arm in `query_hybrid_with_vector_storage`) addresses the symptom for all chunks at once and is a more robust answer — see hybrid-retrieval section below.

