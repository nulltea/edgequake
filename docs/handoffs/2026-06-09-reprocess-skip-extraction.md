# Handoff: reprocess-with-skip-extraction button (pivot from non-destructive archive)

**Repo:** `edgequake` · branch `edgequake-main` · git root `/home/timo/edgequake`
(Rust workspace lives under `edgequake/crates/...`).

## What changed direction

We were planning a **non-destructive archive** rewrite (keep chunks/embeddings/
entities, exclude archived docs at *retrieval* time across naive/local/global/
hybrid). That plan was fully designed but **abandoned as too big**. The pivot:
**leave archive destructive, and instead add a cheap "reprocess (skip
extraction)" button** so a doc can be made queryable-on-chunks again quickly
without re-running the expensive LLM entity/algorithm/repo stages.

- Abandoned plan (local, not in repo — for reference only):
  `~/.claude/plans/starry-gathering-coral.md`. Do **not** implement it. Its one
  reusable finding: every vector row carries an indexed materialized
  `document_id` column (`eq_*_vectors_doc_id_idx`), and retrieval already has a
  per-doc allow-list (`allowed_document_ids` → `context_filter.rs`).

## The new task

Add a way to **reprocess an existing document with `skip_extraction: true`**
(chunks + embeddings only, no entity/algorithm/repo/table extraction), surfaced
as a button in the UI. Most of the machinery already exists — this is mostly
wiring a flag through the reprocess path + a frontend button.

## What already exists (build on this — do NOT rebuild)

The full `skip_extraction` / chunks-only feature is already implemented and on
`edgequake-main`. See the prior handoff for its complete map:
`docs/handoffs/handoff-skip-extraction-feature.md`. Key pieces:

- `PdfProcessingData.skip_extraction: bool` (`edgequake-tasks/src/types/data.rs`)
  — honored end-to-end: `processor/text_insert.rs`, `processor/pdf_processing.rs`
  guard out the heavy stages; finalize as status `partial` + metadata
  `extraction_skipped: true` (`processor/status_updates.rs::set_extraction_skipped_flag`).
- Per-doc trigger-extraction-later endpoint already exists:
  `POST /documents/{id}/extract` (`handlers/documents/extraction.rs`) and
  workspace `POST /workspaces/{id}/extract-pending`.
- Frontend already has the "Skip extraction" checkbox on upload, a `partial` →
  "Chunks only" amber status badge, and a "Run extraction" doc-actions item
  (`edgequake_webui/src/...`, see prior handoff).

## Where to wire the new button

Reprocess endpoints today **hard-code `skip_extraction: false`**:

- `POST /documents/reprocess` → `reprocess_failed` (`handlers/documents/recovery/
  reprocess.rs`). Takes `ReprocessFailedRequest { document_id?, force, max_documents }`
  (`documents_types`); targets failed/cancelled docs.
- `POST /workspaces/{id}/reprocess-documents` → `reprocess_all_documents`
  (`handlers/workspaces/bulk_ops/reprocess_documents.rs`); builds
  `PdfProcessingData` via the helper in `bulk_ops/mod.rs:197` with the literal
  `skip_extraction: false` ("Reprocess always runs the full pipeline").

Suggested minimal implementation:
1. Add `skip_extraction: bool` (default false) to the reprocess request
   type(s) and thread it into the `PdfProcessingData` builder — replace the
   hard-coded `false` at `bulk_ops/mod.rs:197` with the request value. When true,
   also set the `extraction_skipped` flag on finalize (the processor already does
   this when `skip_extraction` is set).
2. Decide the per-doc entry point: either extend `reprocess_failed` (it already
   accepts `document_id`) to accept the flag, or add a small per-doc
   `POST /documents/{id}/reprocess?skip_extraction=true`. Confirm whether the
   reprocess path cleans existing chunks/graph first
   (`storage_helpers::cleanup_document_graph_data` is imported in reprocess.rs) —
   chunk IDs are deterministic (`{doc_id}-chunk-N`) so re-embed is idempotent.
3. Frontend: add a "Reprocess (chunks only)" item next to the existing "Run
   extraction" action in `document-actions-menu.tsx` (+ API fn in
   `lib/api/edgequake.ts`, type in `types/index.ts`). Reuse the `partial` /
   "Chunks only" badge.

## Open question for the next session

Clarify with the user what the button is *for*, since the archive pivot makes the
intent ambiguous: (a) restore an **unarchived** doc cheaply (chunks-only) without
the heavy stages, or (b) a general "re-embed without re-extracting" action for
any doc. This changes whether it lives on archived docs, all docs, or both.

## Build / deploy / verify

- Per `edgequake/CLAUDE.md`: **always** pass `--features postgres` for
  `edgequake-api`/`-tasks`/`-core`/`-storage`/`-agents`/`-algorithms`/umbrella;
  do **not** pass it to `edgequake-pdf`/`-pipeline`.
- Deploy is local Docker build (NOT the ghcr prebuilt quickstart compose):
  `cd edgequake/docker && docker compose build edgequake && docker compose up -d
  edgequake`; `/health` on :8080 = 200. Postgres container `edgequake-postgres`
  (user/db `edgequake`); SearXNG/Crawl4AI run on host :8888/:11235.
- E2E: take a `completed` doc → reprocess with skip → status `partial` /
  "Chunks only", `extraction_skipped: true`, chunk query returns hits, entity/
  graph search returns none → then `POST /documents/{id}/extract` restores
  entities. MCP `edgequake` tools (`document_status`, `query`,
  `graph_search_entities`) cover this without the UI.

## Context note (unrelated, same session)

Earlier this session two fixes were made to **reference-repo detection** and
**deployed** (image rebuilt, container healthy): mixed-case acronym detection
(`extract_acronym`) and the verifier official/third_party prompt
(`edgequake-agents/src/{web_search/github.rs, repo_detection/verify.rs}`). These
are committed-worthy but **not yet committed** — independent of the archive work.

## Suggested skills for the next session
- **EnterPlanMode** is likely overkill now — this is a small wiring task; just
  confirm the open question, then implement.
- **verify** / **run** — exercise the button in the rebuilt deployment.
- **diagnose** — if reprocess-skip surfaces a runtime bug in the partial-status path.
