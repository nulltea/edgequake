---
title: 'Reference Codebase RAG (Phase 2)'
---

# Reference Codebase RAG

> **A separate code graph + vector index for approved reference repositories, designed for coding-agent workflows — porting, integrating, or validating a paper's reference implementation.**

Phase 2 of the Reference Code GraphRAG extension. Sits on top of Phase 1's human-review flow and reuses its clone volume + code embedder, but its index, query path, and presentation surfaces are entirely separate from the paper-centric RAG that drives normal `/query` answers.

Companion to `reference_code.md`, which covers Phase 0 (repo detection) + Phase 1 (per-algorithm snippet approval + query-time enrichment).

---

## 1. What it does

Phase 1 gives the user a GitHub repo, the paper's algorithms, and an approved snippet per algorithm. That snippet lands in a Jina embedding and appears in paper Q&A answers. Useful for understanding — insufficient for building.

Phase 2 fills the gap. When a code match is approved, the approved repo gets a full indexing pass that produces:

1. A **symbol graph** — every top-level function, class, struct, impl, module from the repo's Rust / Python / TypeScript / C / C++ source. Edges cover `defines`, `calls`, `imports`, `references`, `implements`, `inherits` (per language; some edges skipped where semantics don't cleanly apply).
2. A **chunk vector index** — AST-aware chunks centered on approved-algorithm anchors and their structural neighbourhood, embedded with the same Jina code-embedder Phase 1 uses, stored in a dedicated pgvector table.
3. **Retrieval APIs** — semantic NL → code-chunk search (`POST /reference-codebase/query`) and anchor-centric BFS subgraph walks (`GET /indexes/{id}/graph`). Both exposed as MCP tools (`query_code`, `get_symbol_neighborhood`) so a coding agent can use them during a port.
4. **A "Code Graph" UI tab** — visualises the graph per anchor or whole-repo, with a sigma.js renderer tuned for code.

Everything runs off a shared clone the `code-analyzer` sidecar owns, and every retrieval call is scoped by `(tenant, workspace, document_repo, index)`.

---

## 2. When each surface fires

```
 user approves a code match
        │
        ▼
 reference_codebase_index task enqueued  ◀── auto-enqueue (behind env flag)
        │                                   also reachable via POST /indexes
        ▼
 code-analyzer /snapshot  — clone or reuse on shared volume
        │
        ▼
 indexer walks files → symbols → edges → chunks
        │
        ▼
 Jina embeds every chunk → pgvector
        │
        ▼                   ┌── POST /reference-codebase/query     (NL → chunks)
 index status='complete' ───┼── GET /reference-codebase/.../graph  (symbol BFS)
                            └── Code Graph UI tab + MCP tools
```

Nothing about this flow auto-triggers on paper ingestion or repo detection — the gate is a deliberate **code-match approval**. This prevents burning embedding budget on repos nobody reviewed.

---

## 3. Architecture overview

### Separate index, shared embedder

Phase 2 does not extend Phase 1's `code_artifact_embeddings` table. Five new tables (`reference_codebase_{indexes,files,symbols,edges,chunks}`) and one vector table (`reference_codebase_embeddings`, pgvector 896-d + HNSW cosine) live alongside the Phase 1 surface.

Reasons for the split:

- **Different retrieval semantics.** Phase 1 surfaces snippets to *augment a paper answer*; Phase 2 returns raw chunks to *feed a coding agent*. The ranking, filtering, and response shapes have nothing in common beyond the embedding model.
- **Different lifetimes.** A Phase 1 approval lasts until manual reject. A Phase 2 index is tied to `(repo_url, repo_commit, mode)` — re-indexing replaces it wholesale.
- **Graph precision independence.** Phase 2's graph is built for navigation; Phase 1 has no call graph. Keeping them separate avoids coupling a future SCIP precision pass to Phase 1's review UI.

The shared piece is the **embedder**: both phases go through `JinaEmbedder` with the `nl2code` passage prefix, so a Phase 1 snippet and a Phase 2 chunk for the same function are comparable in embedding space. This matters if the two retrievals ever converge (they don't today).

### Separate query path, explicit opt-in

Phase 2 is never merged into regular paper Q&A. `/query` (the main RAG endpoint) surfaces Phase 1 approved snippets via the enrichment path; it doesn't touch Phase 2. Coding-agent retrieval goes through `/reference-codebase/query` or the MCP tools — explicit callers only.

Rationale: paper readers want prose + code citations; coding agents want ~20 chunks per query. Mixing doubles token budget with misaligned incentives. A `POST /query?include_reference_codebase=true` override is a clean future extension; nothing about the current design precludes it.

### Clone volume ownership

The `code-analyzer` sidecar owns the `/workspace` volume read-write (clones repos, mutates on `/analyze` and `/snapshot`). EdgeQuake mounts the same volume **read-only** (`code-analyzer-workspace:/workspace:ro`) and indexes from it directly — no copy, no intermediate tarball. When EdgeQuake needs a clone that doesn't exist yet, it calls `POST /snapshot` on the analyzer rather than re-cloning on its own side. Single owner for disk state.

---

## 4. Key design decisions

**Separate index vs. extension of Phase 1.** Separate. See §3 above. A shared vector store with `source_type: "phase2_chunk"` metadata would have saved a migration but forced every Phase 1 retrieval to filter by metadata to exclude Phase 2 chunks — ongoing cost for a one-time saving.

**Tree-sitter via `ast-grep-core`, not raw `tree_sitter::Query`.** The plan called for tree-sitter; the question was whether to write cursor walks by hand or go through a higher-level library. `ast-grep-core` wraps the same grammars and gives us a declarative pattern DSL, which means a new language is a few rules, not a bespoke walker. Per-language work drops to ~30 lines each for Rust / Python / TypeScript / C / C++. The one thing ast-grep can't express — reference resolution ("this name binds to a known symbol in scope") — we do via a direct `tree_sitter::Query` pass over the same tree, so there's no redundant parse.

**Edge taxonomy scoped per language, not maximal.** Some edges don't mean the same thing everywhere. `implements` is well-defined for Rust (`impl Trait for Type`) and TypeScript (`class X implements Y`); for Python it's duck typing and emitting it would mislead. `inherits` works for Python (`class X(Y)`) and TypeScript (`extends`), doesn't for Rust (no class inheritance). C/C++ `implements` + `inherits` need type resolution across templates and multiple inheritance to be correct — deferred to Phase 3 via SCIP. The current taxonomy avoids faking edges the code can't actually guarantee.

| Edge | Rust | Python | TypeScript | C | C++ |
|---|---|---|---|---|---|
| `defines` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `calls` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `imports` | `use` | `import` | `import`/`require` | `#include` | `#include` |
| `references` | ✓ | ✓ | ✓ | — | — |
| `implements` | `impl Trait for T` | — | `implements` | — | deferred |
| `inherits` | — | `class X(Y)` | `extends` | — | deferred |

**Graph in SQL, not Apache AGE.** Phase 2's edge count per index is typically a few thousand (V3DB is 6109). AGE's Cypher tooling is a good fit for knowledge-graph-scale queries but adds operational weight for essentially "recursive CTE over a flat edge table." A single `reference_codebase_edges` table with composite indexes gives us BFS at ~50 ms for V3DB-scale repos. The AGE graph stays for Phase 1's `HAS_REFERENCE_IMPL` algorithm → code_function edges, which are sparse and cross-reference paper entities.

**Anchor-centric retrieval, with a "whole repo" escape hatch.** Default graph view is N-hop BFS around an approved-algorithm anchor — matches the porting use case ("show me what this function calls and what calls it"). A `?whole=true` toggle returns the top-N symbols by degree plus every symbol↔symbol edge between them, for lay-of-the-land exploration. Implemented as two branches in the same endpoint; the UI exposes both via a checkbox.

**Dynamic hops, bounded by real graph diameter.** The slider max is the index's approximate graph diameter, computed on-demand per `GET /indexes/{id}` call via a recursive CTE BFS from the highest-degree symbol. Computing exact graph diameter (max shortest path across all pairs) is O(V·(V+E)) — too costly. Eccentricity from the top-degree seed is a close lower bound in practice, and runs in ~50 ms on a 6k-edge graph. For V3DB this returns 9, reaching 291 of 392 symbols (the other 101 are in disconnected fragments — test helpers, isolated utility files).

**Auto-enqueue behind a flag, idempotent.** On approval, `EDGEQUAKE_REFERENCE_CODEBASE_AUTO_INDEX=true` fires a fire-and-forget `tokio::spawn` that does an `INSERT … ON CONFLICT DO NOTHING RETURNING id` against the `(tenant, workspace, document_repo, commit, mode)` unique key. Sibling approvals on the same repo enqueue exactly one task. The HTTP response doesn't wait on the clone. Flag defaults off — manual `POST /indexes` stays the only trigger until operators opt in.

**Auto-triggered caps.** Rows inserted by auto-enqueue get `auto_triggered = TRUE` and a lower `max_files_override` (~5000) than the global cap. Protects the Jina quota against a click-approval on a 20k-file repo. Explicit `POST /indexes` with `force_reindex=true` lifts it.

**`code-analyzer` owns cloning, even for Phase 2.** Phase 2 needs the repo on disk but doesn't need Claude. Rather than teach EdgeQuake to clone on its own side (and risk divergent repo paths), the analyzer got a new `POST /snapshot` endpoint — same clone semantics as `/analyze`, minus the Claude Code invocation. EdgeQuake calls it before indexing. Single clone owner, same `/workspace/<sha>` path convention, no analyzer-vs-edgequake concurrency on disk.

---

## 5. Trade-offs consciously accepted

**Same repo, N papers, N indexes.** If two papers both cite the same repo, we build two indexes — different `document_repo_id`s force different rows through the unique key. Wasteful (double clone, double embedding spend) but correct per tenant/workspace scoping. Dedup on `(repo_url, repo_commit)` is a Phase 3 follow-up; punting today because V3DB-scale papers are so small the waste is a handful of cents.

**`calls` edges to unresolved external names are stored but not rendered.** V3DB's 5261 `calls` edges include ~4700 references to `numpy.array`, `torch.zeros`, etc. — target symbol isn't in this repo. We keep them in `reference_codebase_edges` with `target_symbol_id = NULL` and `target_name` populated, so MCP `get_symbol_neighborhood` can filter by `target_name` and the stats on the index are accurate. The graph endpoint filters to edges where *both* endpoints are known symbols — the subgraph only shows internal calls (569 of 5261 for V3DB). Two surfaces, two filters, same storage.

**Preprocessor macros are not expanded.** C and C++ source goes through tree-sitter raw. Macro-hidden calls (`#define FOO(x) bar(x)`) don't resolve to `bar`. Acceptable approximation — codemem makes the same choice. SCIP integration in Phase 3 fixes this.

**Python `implements` / `inherits`.** Duck typing makes `implements` meaningless in Python — not emitted. `inherits` is inferred purely from `class X(Y):` syntax; conditional superclasses (`class X(Y if COND else Z):`) are best-effort. Good enough for retrieval; accurate type ancestry is a Phase 3 SCIP job.

**Silent parse failures.** Tree-sitter can crash on exotic macro expansion in Rust, new Python match statements on old grammars, etc. The indexer catches per-file and increments `reference_codebase_files.parse_errors` — graph looks thin rather than crashing the whole index. A UI banner surfacing `parse_errors > 5% of file_count` is TODO; the counter is populated today.

**Clone storage never GCs automatically.** Repos persist on the shared volume until the user explicitly re-indexes. For a ~250 MB cap per repo and dozens of repos per workspace, disk pressure is a real risk in production. A `EDGEQUAKE_REFERENCE_CODEBASE_CLONE_TTL_DAYS` sweeper is specced but not implemented.

---

## 6. Retrieval shape

### Semantic search (`POST /reference-codebase/query`)

Embed the NL query with Jina `nl2code` query prefix, HNSW cosine search against `reference_codebase_embeddings`, join to chunks + index for presentation fields. Filters stack multiplicatively:

- `document_repo_id` — restrict to one paper's repo.
- `index_id` — restrict to one build (tighter; useful for reproducibility).
- `algorithm_ids[]` — restrict to the indexed code neighborhood connected to approved algorithms, not just the exact anchor chunks.
- `limit`, `max_distance` — cap hits + similarity ceiling.

Result set rank boosts anchor-focus:

```sql
ORDER BY algorithm_focus * -0.08 + cosine_distance ASC, cosine_distance ASC
```

An anchor chunk at distance 0.45 beats a plain symbol at 0.42 — makes reviewer-approved hits win ties. Weight chosen small (0.08) so distant anchors don't outrank much closer non-anchor chunks.

### Subgraph walk (`GET /indexes/{id}/graph`)

Seed resolution, cheapest-first:

1. `anchor_artifact_id` — resolve approved `code_artifact` → overlapping symbols (file + line-range overlap). Returns every matching symbol (an artifact can resolve to multiple if the reviewer's line range spans several).
2. `anchor_symbol` — exact name match, case-insensitive fallback. Used by the MCP `get_symbol_neighborhood` tool when the agent already has a symbol name from a `query_code` result.
3. Neither — top-N by degree. The "lay of the land" fallback for `mode='full'` indexes or unfamiliar repos.

BFS via recursive CTE bounded by `hops` and `max_nodes`. `?whole=true` skips BFS, returns top-N seeds plus every symbol↔symbol edge between them.

Each returned node carries `chunk_id` when a matching chunk exists — one hop from graph click to chunk content in the UI, no second round-trip. Anchor symbols get `is_anchor = true` so renderers can style them.

### MCP tools

`query_code` — wraps semantic search, renders markdown with fenced code blocks, file:line ranges, and GitHub deep-links pinned to `repo_commit`. Description nudges the agent toward porting / integration use (not paper questions).

`get_symbol_neighborhood` — wraps the graph endpoint with `anchor_symbol` / `anchor_artifact_id`. Renders the subgraph as per-symbol outgoing-edge lists, anchor markers (⭐) highlighting reviewer-approved symbols.

Both pull tenant/workspace from the MCP server's env-bound singleton, never from tool arguments — multi-tenant leak guard.

---

## 7. Tools and services

External dependencies introduced by Phase 2:

| Tool | Purpose | Why chosen |
|---|---|---|
| **ast-grep-core** (v0.42) | Declarative pattern matching over tree-sitter AST | 30-line rule sets per language vs. hand-written `tree_sitter::Query` walks. Inherits codemem's rule taxonomy for free (and means we can cross-diff when rules misbehave). |
| **ast-grep-language** (v0.42) | Bundled grammar loader — pulls `tree-sitter-{rust,python,typescript,c,cpp,go}` transitively | One dep for five languages. Saves re-declaring grammar crates + feature juggling. |
| **tree-sitter-{c,cpp,python,rust,typescript}** | Actual grammars | Via ast-grep-language. Battle-tested, version-pinned at ast-grep's snapshot. |
| **Jina code-embeddings 0.5B** (reused from Phase 1) | Chunk embedding + query embedding | Same 896-d embedder, same `nl2code` prefix family. Phase 1 and Phase 2 embeddings are comparable. |
| **pgvector** + HNSW (reused) | Cosine-similarity index on `reference_codebase_embeddings` | Same extension EdgeQuake uses everywhere else. HNSW handles 20k+ chunks per workspace without tuning. |
| **Recursive CTE (Postgres)** | BFS subgraph + graph diameter estimation | Single SQL query, 50 ms on V3DB (6k edges). Avoids a whole graph-DB dependency for what's fundamentally "walk this small edge table." |
| **sigma.js + graphology** (reused) | Code graph rendering | Same libraries the paper knowledge graph uses. Wrapped in a dedicated `CodeGraphRenderer` for code-specific colouring (hue per language, anchors brighter) — trying to share the entity renderer would have meant grafting kind-aware overrides onto every signal. |
| **Model Context Protocol** (Anthropic) | `query_code` / `get_symbol_neighborhood` tool surface for coding agents | Standard agent protocol. Reuses the existing EdgeQuake MCP server — no new transport layer. |

---

## 8. Where the pieces live

Services:

- `code-analyzer/` — gets `POST /snapshot` endpoint alongside existing `/analyze`. Owns the shared workspace volume.
- `mcp/src/tools/reference-codebase.ts` — both MCP tools in one file.
- `sdks/typescript/src/resources/reference-codebase.ts` — typed client for `createIndex`, `getIndex`, `query`, `graph`.

Rust crates:

- `edgequake-agents/src/reference_codebase/` — the indexer (scan + symbols + edges + chunker + embed), storage trait + Postgres adapter. All of the heavy lifting.
- `edgequake-agents/src/reference_codebase/treesitter.rs` — per-language extraction via ast-grep-core + tree-sitter.
- `edgequake-api/src/handlers/reference_codebase/` — HTTP endpoints (`/indexes`, `/indexes/{id}/graph`, `/query`, `/by-repo/{id}`).
- `edgequake-api/src/processor/reference_codebase.rs` — task handler wiring snapshot → index → chunk → embed.
- `edgequake-api/src/handlers/code_reference/mod.rs` — auto-enqueue side-effect in the approval path (behind `EDGEQUAKE_REFERENCE_CODEBASE_AUTO_INDEX`).
- `edgequake-storage/src/traits/reference_codebase_vector.rs` + Postgres adapter — vector search trait.

WebUI:

- `edgequake_webui/src/components/code-graph/` — Code Graph tab + sigma.js renderer.
- `edgequake_webui/src/types/reference-codebase.ts` — DTOs mirroring the Rust API shape.

Migrations:

- `045_add_reference_codebase_indexes.sql` — indexes/files/symbols/edges/chunks tables.
- `046_add_reference_codebase_embeddings.sql` — pgvector + HNSW.
- `048_reference_codebase_auto_trigger.sql` — `auto_triggered`, `max_files_override`, `parse_errors` columns.

Environment (defaults shown):

```
EDGEQUAKE_REFERENCE_CODEBASE_INDEXING=off       # outer kill-switch
EDGEQUAKE_REFERENCE_CODEBASE_AUTO_INDEX=false   # auto-enqueue on approval
EDGEQUAKE_REFERENCE_CODEBASE_AUTO_MAX_FILES=5000
EDGEQUAKE_REFERENCE_CODEBASE_MODE=algorithm_focused
EDGEQUAKE_REFERENCE_CODEBASE_MAX_REPO_MB=500
EDGEQUAKE_REFERENCE_CODEBASE_MAX_FILES=25000
EDGEQUAKE_REFERENCE_CODEBASE_MAX_FILE_BYTES=1048576
EDGEQUAKE_REFERENCE_CODEBASE_MAX_CHUNKS=20000
EDGEQUAKE_REFERENCE_CODEBASE_QUERY_LIMIT=12
EDGEQUAKE_REFERENCE_CODEBASE_GRAPH_HOPS=1       # declared, unused (slider drives)
EDGEQUAKE_REFERENCE_CODEBASE_SCIP=off           # declared, Phase 3
```

---

## 9. Not in Phase 2

Deliberately deferred:

- **SCIP precision layer** (flag declared, unwired). Unlocks accurate `implements` / `inherits` for C++ and type-aware `references` for C / C++. Adds per-language indexers as runtime deps (gopls, pyright, scip-rust, scip-cpp) — worth doing once the tree-sitter approximation starts hurting on real port tasks.
- **Cross-paper repo dedup** on `(repo_url, repo_commit)`.
- **Clone storage TTL sweeper.**
- **Parse-error UI banner.** The counter is populated; the banner needs a `SUM(parse_errors)` aggregate on the index response — simple wire-up, not done yet.
- **Whole-repo mode default.** Anchor-centric with a toggle is the right trade-off while repos stay small; revisit if users complain.
- **Phase 2 + paper-query fusion.** Explicitly separate today. A `POST /query?include_reference_codebase=true` flag would merge Phase 2 chunks into paper-query context. Easy to add; waiting for a concrete use case.

---

## 10. Verification

Live verification against V3DB zk-ivf-pq after a full re-index with the tree-sitter extractor:

```
symbols:     392    (384 function + 6 class + 2 type)
files:       88     (51 python + 37 rust)
edges:       6109   (5261 calls + 456 imports + 392 defines)
chunks:      1      (algorithm_focused — the approved anchor only)
max_depth:   9      (BFS from top-degree seed, reaches 291/392 symbols)
```

The 1:16 symbol:edge ratio is normal (~13 calls per symbol average). `max_depth=9` reaching 291 of 392 means the connected component anchored by the `main`/`bench` entry points covers most of the indexed surface; the remaining 101 are isolated test helpers not reachable from the main call graph — expected for bench-heavy code.

MCP e2e tests (`mcp/tests/e2e/reference-codebase.test.ts`) exercise both tools against the live endpoint — 2/2 green.
