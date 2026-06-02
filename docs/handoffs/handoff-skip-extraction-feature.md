# Handoff: chunks-only upload + trigger-extraction-later

**For:** an agent on the USER'S DESKTOP (where EdgeQuake is deployed). The feature was authored on a laptop and has been **committed + pushed to `edgequake-main`** (rebased onto the desktop's `b6c0ef6e`, conflicts already resolved on the laptop). The desktop's job is **pull → build → recreate containers → e2e test** — no git conflict resolution should be needed.

**Repo:** `edgequake` · `origin` = https://github.com/nulltea/edgequake · branch `edgequake-main`

---

## The task

Let users upload documents **without entity extraction** — convert PDF→Markdown, extract figures/tables, embed + save chunks, so a paper is queryable against chunk embeddings only. Then let them **trigger the heavy LLM stages later**, per-document and workspace-wide (entity/relationship extraction, algorithm extraction, reference-repo detection, table classification).

Deliverables (all implemented):
1. Backend chunks-only ingestion.
2. "Skip extraction" checkbox on the upload dropzone (default off).
3. Trigger extraction afterward — per-doc action + workspace bulk button.

---

## What's done

Feature commit on `edgequake-main` (see `git log`/`git show` for the exact diff — not duplicated here). Verified before push: `cargo check -p edgequake-tasks -p edgequake-api --features postgres` clean (post-rebase), frontend `tsc --noEmit` zero source-file errors. **No runtime/e2e has been run yet.**

Backend (`edgequake/crates/...`):
- `edgequake-tasks/src/types/data.rs` — `skip_extraction: bool` on `PdfProcessingData`.
- `processor/text_insert.rs` — when skipping, stop after phase-1 (chunks+embeddings persisted → queryable), set status `partial` + `extraction_skipped: true`.
- `processor/pdf_processing.rs` — keep OCR + figure/table **media** backfill + chunk embed; guard out algorithm Pass 2+3, repo detection, figure-entity linking, table classification; finalize as `partial`.
- `processor/status_updates.rs` — `partial` status mapping + `set_extraction_skipped_flag` helper.
- Upload handlers accept the flag: `documents_types/upload.rs`, `documents/upload/text_upload.rs`, `pdf_upload/{types,upload,helpers}.rs`; reprocess paths set `false`.
- **NEW** `handlers/documents/extraction.rs` — `POST /documents/{id}/extract` + shared `queue_extraction_tasks` (reuses existing `Insert`/`AlgorithmExtraction`/`RepoDetection`/`TableClassification` task types; **no PDF re-OCR** — algorithm extraction re-detects layout from stored PDF bytes; PDF entities re-run over `pdf_documents.markdown_content`).
- **NEW** `handlers/workspaces/bulk_ops/extract_pending_documents.rs` — `POST /workspaces/{id}/extract-pending`.
- `routes.rs` + the two `mod.rs` files register the new routes/handlers; `DocumentInfo` gained `extraction_skipped`.

Frontend (`edgequake_webui/src/...`):
- `types/index.ts` — `skip_extraction?` on requests, `extraction_skipped?` on `Document`.
- `components/documents/document-dropzone.tsx` — "Skip extraction" checkbox; wired via toolbar + manager + `hooks/use-file-upload.ts` + `lib/api/edgequake.ts`.
- Trigger UI: "Run extraction" item in `document-actions-menu.tsx` (threaded through table-section/row); workspace "Extract pending" banner+button; `triggerExtraction` / `extractPendingDocuments` API fns.
- `status-badge.tsx` — new `partial` → "Chunks only" badge (amber).

Git integration done on the laptop: feature commit rebased onto `b6c0ef6e`; the only overlaps were `documents/mod.rs` (module registration) and `routes.rs` (route ordering), both additive and resolved.

---

## What's left (desktop)

1. **Pull** `edgequake-main` (should fast-forward; no conflicts expected).
2. **Build + recreate containers.** The repo's `docker-compose.quickstart.yml` pulls prebuilt ghcr.io images (`image:`, no `build:`) — that will NOT include this code. Use the desktop's local-build path (check `Makefile` targets like `backend-build`, any `docker/` build compose, or Dockerfiles). Build backend (`--features postgres`) + frontend, recreate `edgequake-api` and `edgequake-frontend`.
3. **E2E test:**
   - Upload a small **markdown** doc with `skip_extraction: true` → status `partial` / "Chunks only", `extraction_skipped: true`; chunk query returns hits; entity/graph search returns none.
   - Upload a **PDF** with skip → markdown + figures/tables present, no algorithms, status `partial`.
   - `POST /documents/{id}/extract` (or the "Run extraction" menu item) → entities (PDF: + algorithms/tables/repos) appear; flag clears; status → `completed`/`partial_failure`.
   - Workspace **"Extract pending"** button / `POST /workspaces/{id}/extract-pending` → all flagged docs queued.
   - MCP `edgequake` tools (`document_upload`, `document_status`, `query`, `graph_search_entities`) cover steps without the UI.

---

## Watch-outs (by design, v1)
- On trigger, the 4 heavy tasks run concurrently; status reflects entity-extraction completion while algorithm/table/repo finish independently (doc stays queryable). `extraction_skipped` is cleared up front so the doc leaves the "pending" set immediately.
- PDF entity re-extraction re-chunks/re-embeds from stored markdown; chunk IDs are deterministic (`{doc_id}-chunk-N`) → idempotent (no duplicate chunks).
- Per `edgequake/CLAUDE.md`: **always** pass `--features postgres` for `edgequake-api`/`-tasks`/`-core`/`-storage`/`-agents`/`-algorithms`/umbrella; do NOT pass it to `edgequake-pdf`/`-pipeline`.

## Suggested skills for the desktop agent
- **verify** or **run** — launch the rebuilt app and confirm the feature in the real deployment.
- **diagnose** — if the e2e surfaces a runtime bug in the skip/trigger paths.
- **Git Workflow Master** — only if the pull unexpectedly conflicts.
