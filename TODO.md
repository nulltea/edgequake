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
