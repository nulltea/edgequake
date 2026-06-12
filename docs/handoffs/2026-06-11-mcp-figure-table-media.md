# Handoff — MCP figure/table media in `query` & `document_get_md`

**Date:** 2026-06-11
**Branch:** `feat/document-references`
**Status:** WIP, **uncommitted**. Figure-image plumbing works end-to-end; one
known bug remains in the `query` path and a design decision on tables is open.

## Goal

Make the MCP read paths surface paper **figures** (and clarify **tables**) so
an agent can actually see them, instead of opaque `![fig_X](edgequake-figure)`
sentinels. Example doc: XorMM, `b4ba9edc-4e23-44e3-9eed-9470257e5ea6`, in
workspace `00000000-0000-0000-0000-000000000003` (tenant `…0002`).

## What was built (all compiling, partially shipped to the running stack)

Changes span four areas — see `git diff` for specifics:

1. **Storage — `crates/edgequake-pdf/src/backend/figure_extract.rs` + `mod.rs`**
   Figures now encode as **lossy WEBP q80** (was PNG). `ExtractedFigure.png_bytes`
   renamed `image_bytes`; mime `image/webp`. Adds `webp = "0.3"` dep. Unit test
   covers encode + decode round-trip (`cargo test -p edgequake-pdf --lib figure_extract::`).

2. **API — `crates/edgequake-api/src/handlers/documents/query/figure_media.rs`**
   `GET /documents/{id}/figures/{figure_id}` now **always serves WEBP**: stored-WEBP
   streams as-is, legacy PNG transcodes on read (`transcode_to_webp`, q80). Adds
   `image` + `webp` deps. Unit tests written but **can't run** — blocked by the
   pre-existing `enable_rerank`/`chunk_min_score` fixture breakage in the
   `edgequake-api` test binary (documented in `edgequake/CLAUDE.md`). Verified live
   instead: `fig_12_0` 41 KB PNG → 11.6 KB WEBP, `Content-Type: image/webp`.

3. **TS SDK — `sdks/typescript/src/resources/documents.ts` + `types/query.ts`**
   Added `documents.getFigureMedia(docId, figureId): Promise<Blob>` (always WEBP).
   Added `kind` + `figure_id` to `SourceReference` (the type had drifted behind the
   Rust struct).

4. **MCP — `mcp/src/tools/figures.ts` (new), `document.ts`, `query.ts`**
   - `document_get_md`: **rewrites `![id](edgequake-figure)` → fetchable media URL**
     (`rewriteFigureSentinelsToUrls`). Per user instruction "for Markdown
     specifically, inline URLs." Verified live: 0 sentinels remain, 8 URLs emitted,
     URL fetch returns WEBP.
   - `query`: inlines retrieved figure chunks as **base64 WEBP image blocks** via
     `fetchFigureBlocks`. Tests: `mcp/tests/document-get-md.test.ts` (4 tests, green).

**Docker:** the `edgequake` container (compose project `docker`, file
`edgequake/docker/docker-compose.yml`) was **rebuilt + restarted** with layers
1–3, so the live API already serves WEBP. The MCP server must be reconnected
(`/mcp`) after any `mcp/` rebuild for changes to take effect in-session.

## OPEN BUG — `query` images at wrong position + truncated

`query.ts` builds one big text block then **appends image blocks at the array
tail**. The harness (Claude Code, not MCP/server) caps tool results at
~**25,000 tokens** (`MAX_MCP_OUTPUT_TOKENS`, unset → default). The trailing
image is the first thing truncated, so the user sees caption text but no image.

**Fix #1 (agreed direction pending user pick):** rebuild `query`'s response as an
**interleaved content array** — emit each chunk's text, and for a `kind==='figure'`
chunk emit its base64 image **immediately after that chunk's caption line**, not
at the end. Also trim what `query` dumps (cap entities/relationships, shorter
snippets) so blocks fit under the cap.

## Findings on TABLES (by design, with a gap)

- Table chunks (`kind='table'`, `table_html`/`table_rows`) are **NOT vector-indexed**
  (`pdf_processing.rs:921-930`) → **never retrievable via `query`** (0 in both probes).
- Table numbers are made searchable by **inlining the table as GFM into the
  neighbouring text chunk** before embedding. So tables surface in `query` only as
  GFM text inside a retrieved *text* chunk.
- `SourceReference` exposes `kind`/`figure_id`/`caption` but **no table fields**.
- `document_get_md` *does* render tables: `inline_tables=true` rehydrates
  `![tbl_…]` → GFM via DB lookup (`pdf_upload/content.rs`). Confirmed working.
- **Tables are never images.** The `Table N`/`Figure N` `<div>` lines the user saw
  are caption text in retrieved prose, not media.

**Fix #3 (open decision):** to get tables into `query` reliably needs (a) expose
`table_html`/`table_rows` on `SourceReference` + render in MCP, AND likely (b)
vector-index table chunks (bigger change). User hasn't chosen scope.

## Live-data reference (correct workspace `…0003`, hybrid, context_only)

- `"Figure 5: Query Time With Volumes"` → 8 chunks (6 figure w/ figure_id, 2 text), 0 tables.
- `"XorMM query time… (Figure 5)"` → 12 chunks (1 figure, 11 text), 0 tables.
- Figure chunks are embedded by **caption**, so they only rank in when the query
  matches the caption — broad caption query surfaces more figures than scoped prose.

## Next steps

1. Confirm with user: implement **Fix #1** (interleave base64 at caption position
   in `query` + trim payload). Prove with a deterministic vitest before claiming.
2. Decide scope on **Fix #3** (tables in query) — likely defer; it's a pipeline change.
3. Then **commit** — group as: (a) `edgequake-pdf` WEBP storage, (b) `edgequake-api`
   WEBP endpoint, (c) SDK+MCP. Nothing committed yet. Don't reprocess any docs
   (legacy figures transcode on read).
4. Pre-existing cleanup (separate): fix the `enable_rerank`/`chunk_min_score`
   `edgequake-api` test fixtures so the figure_media transcode unit tests can run.

## Process notes for next agent

- **Verify against ground truth, don't trust the harness render.** I twice claimed
  "works" based on my own tool-result view; the user's client truncated the tail.
  Reproduce via direct API/SDK probes and deterministic tests.
- **Always pass the correct workspace** (`…0003` for XorMM) when probing `query` —
  wrong workspace returns empty/garbage and misleads.
- Memory: query figures/results by **scheme name** (e.g. XorMM), not figure label
  alone (`feedback_query_scope_by_scheme`). Don't reprocess without permission.
  Don't auto-commit.

## Suggested skills for next session

- `/diagnose` — for the remaining `query` interleave/truncation work (build the
  deterministic test as the feedback loop).
- `/verify` — to confirm behaviour live after the MCP rebuild + `/mcp` reconnect.
