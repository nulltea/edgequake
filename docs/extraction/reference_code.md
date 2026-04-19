---
title: 'Reference Code GraphRAG'
---

# Reference Code GraphRAG

> **How EdgeQuake links every extracted algorithm to the actual code that implements it, under human review, and surfaces both in one answer at query time.**

Shipped on the `code-graph` branch. Phase 0 (repo detection) and Phase 1 (algorithm → code localization, approval, embedding, query enrichment) are live. Phase 2 (full reference-codebase RAG for coding-agent retrieval) is implemented behind explicit API/task entrypoints.

---

## 1. Why this exists

EdgeQuake already extracts algorithms from papers as reviewable, structured rows (name, steps, pseudocode). At query time, retrieval returns the paper's **prose** — the author's description of what the algorithm does.

The gold-standard implementation — the actual Rust/Python/C++ code that realises the algorithm, typically in the paper's companion GitHub repo — was invisible to EdgeQuake. A user asking *"how does the rebalancing algorithm work?"* got the paper's paragraph about it, not the function that runs it.

This feature closes that gap. For each approved algorithm, EdgeQuake:

1. Finds the paper's reference repo.
2. Locates the implementing function.
3. Asks a human reviewer to confirm the match.
4. On approval, embeds the snippet and attaches it to the algorithm in the knowledge graph.
5. At query time, surfaces the approved snippet next to the paper prose so the LLM answers from *both* sources.

The result: answers stop paraphrasing the abstract and start citing the code — with a GitHub deep-link to the exact commit the reviewer saw.

---

## 2. End-to-end user flow

```
 paper upload
      │
      ▼
 repo detected ───────────────────────────┐
      │ (abstract link / web search)     │ user sees candidate
      ▼                                   │ on document page,
 user approves repo                       │ clicks "Use this repo"
      │                                   │
      ▼                                   │
 code-analyzer clones repo, ◀─────────────┘
 localizes each approved algorithm
      │
      ▼
 candidate snippets appear in
 "Code Matches" tab (pending)
      │
      ▼
 user reviews each match ──────────────┐
      │                                │ approves → snippet is
      │ rejects → nothing happens     │ embedded + edged into graph
      ▼                                │
  (no side effects)                    │
                                       ▼
                              ┌────────────────────┐
                              │  query time:       │
                              │  paper prose +     │
                              │  approved code     │
                              │  in one answer     │
                              └────────────────────┘
```

Nothing in this flow is automatic past ingestion — every step that adds content to the knowledge base (accepting a repo, accepting a code match) is a deliberate human action. This is the same review posture EdgeQuake already has for extracted entities and algorithms.

---

## 3. Finding the repo (Phase 0)

A paper rarely makes its reference implementation easy to find. Three detection strategies run cheapest-first; the first that hits wins.

### Strategy 1 — PDF link annotations

Modern papers embed clickable hyperlinks in the PDF itself, as `Link` annotations with `URI` actions. No OCR, no LLM, no regex. The PDF parser walks these directly and filters by host allow-list (`github.com | gitlab.com | bitbucket.org`).

**The bibliography problem.** Papers cite *other* repos in their references — we want only the paper's *own* implementation. Two filters suppress the noise:

- **Physical position.** Find the page/y-coordinate where a heading matching `^(References|Bibliography|Works Cited)\s*$` starts. Discard URLs that appear below it.
- **Section scoring.** Rank surviving URLs by which section contains them: Abstract / Intro / "Code & Data Availability" scores highest, mid-body next, ambiguous last.

Catches ~90 % of modern arXiv preprints at essentially zero cost.

### Strategy 2 — Web search fallback

When strategy 1 finds nothing (scanned PDFs, old preprints, non-hyperlinked PDFs), fall back to the user's self-hosted **SearXNG** with a query like `"{first_author} {paper_title} github"`. The top results are scraped with **Crawl4AI**, and a single small LLM prompt extracts a canonical `github.com/{org}/{repo}` URL from the cleaned page text.

Only runs when strategy 1 failed. Skipped entirely if `SEARXNG_URL` is unset — no hard dependency.

### Strategy 3 — entity-extraction safety net

A `CODE_REPOSITORY` entity type was added to the default entity-extraction pass. Any repo URL the regular entity extractor notices becomes a first-class graph node. Zero extra LLM calls (piggybacks on the pass that was running anyway), and makes repos queryable on their own terms (e.g., "what other papers cite this repo?").

### Why not Papers with Code

The obvious candidate for paper → GitHub lookup, but paperswithcode.com shut down in July 2024. Out of reach.

### User in the loop

Even high-confidence detections sit in `pending` status. The document detail page shows the candidate(s); the user picks the right one or adds a repo URL by hand. Nothing downstream fires until a repo is approved.

---

## 4. Localizing the algorithm in the repo (Phase 1)

Once a repo is approved, EdgeQuake has to find the function(s) that implement each algorithm. This is a *non-trivial retrieval problem*: algorithm names in the paper rarely match function names in the code, comments are sparse, and the implementation may be spread across helpers.

### The decision: drive Claude Code in headless mode

Three options were considered:

| Option | What it is | Verdict |
|---|---|---|
| **Claude Code CLI, headless** | `claude -p "find X in this repo" --allowed-tools Read,Grep,Glob --output-format json` | **Picked.** Read-only sandbox matches the task shape exactly. Subscription-auth friendly. Structured JSON output is validated server-side. |
| Bespoke retriever + LLM rerank | tree-sitter chunking + BM25 + embedding search + LLM re-rank | Cheapest per-run and fully deterministic, but requires building and tuning per-language pipelines. Kept in mind as a Phase-2 fallback if cost or privacy pressure shows up. |
| Full autonomous agent (OpenHands, swiftide-agents) | ReAct loop with unrestricted tool use | Over-spec'd. Requires Docker-in-Docker sandbox. Over-fitted to SWE-bench-scale editing, not to pin-point retrieval. |

The Claude Code CLI wins because it's essentially a read-only retrieval agent we don't have to build, sandbox, or maintain. `Grep` on likely names, `Read` on bodies, returns a JSON with `(file, start_line, end_line, rationale)`.

### Why the CLI, not the Python Agent SDK

Anthropic's Python Agent SDK is documented as API-key auth only. The user has a Claude Code subscription, so we want to honour that for cost reasons — and the supported path for subscription auth in headless contexts is `CLAUDE_CODE_OAUTH_TOKEN`, a one-year token minted via `claude setup-token`. The SDK doesn't consistently honour this token. The CLI does, and since the SDK shells out to the same binary anyway, skipping the SDK loses no capability and avoids a layer of "unsupported" warnings.

### How the sandbox actually works

Everything the agent does is bounded so that cloned repo code can't hurt anything:

- **Tool allow-list.** `Read, Grep, Glob` only. No `Edit`, no `Bash`, no `Write`.
- **Non-root container.** uid 10001, owns the workspace volume, read-only to the rest of the filesystem.
- **Per-run config isolation.** Each request mints a fresh `CLAUDE_CONFIG_DIR=/tmp/claude-run-<uuid>` so concurrent analyses don't corrupt shared plugin cache or session state.
- **Env scrubbing.** `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` are stripped before spawning the CLI. Otherwise API-key auth would silently take precedence over subscription auth.
- **JSON schema contract.** The CLI is asked for a specific output shape; pydantic re-validates on return. Paths outside the repo root and non-existent files are dropped before persistence.

### Model choice

Sonnet by default. Haiku is ~4–5× cheaper and fast enough for grep-driven localization on easy cases, but misses harder matches (e.g. inlined or renamed algorithms). Opus is overkill for retrieval and burns the user's Opus quota — we pin Sonnet explicitly via `CODE_ANALYZER_DEFAULT_MODEL` so routine jobs don't accidentally draw from the wrong bucket. Per-request override is still possible.

Observed cost: $0.08–$0.40 per paper on Sonnet for 5–10 algorithms. Batched into one session per paper so there's one context load for N algorithms rather than N cold starts.

### Clone strategy

- `git clone --depth=1 --filter=blob:none` — shallow + blobless.
- Size cap at 500 MB by default. Repos over the cap fail fast with a clear error rather than silently truncating.
- License detected via filename match on `LICENSE | LICENSE.md | COPYING`.
- The clone *persists* on a shared volume after localization completes, so edgequake can read snippets by `(file, start, end)` without re-cloning. A nightly GC trims clones older than N days (not yet enabled in Phase 1).

### A separate container

The code-analyzer runs in its own Docker container, not linked into edgequake. Reasons:

1. **Credential blast radius.** The Claude subscription token lives only in that container. The main binary never sees it.
2. **Backend swap surface.** If Phase 2 builds a bespoke retriever, we swap one container without touching edgequake. The edgequake side talks to a thin HTTP boundary (`POST /analyze`).
3. **Volume ownership.** The analyzer owns the clone volume (read-write). Edgequake mounts it read-only.

---

## 5. Human review

The analyzer's findings land as `pending` `code_artifacts` rows, not authoritative data. A new **Code Matches** tab on the document page groups candidates by algorithm and shows, for each:

- Syntax-highlighted snippet.
- `file:start-end` reference.
- The reviewer rationale the model produced (*"rebalance_clusters implements the capacity-constrained rebalancing algorithm because it loops while any cluster exceeds cluster_bound..."*).
- Confidence badge (high / medium / low), status badge (pending / approved / rejected).
- Direct link to `github.com/{org}/{repo}/blob/{commit}#L{start}-L{end}` at the pinned commit the analyzer saw.

Approval is the gate. Until a reviewer hits **approve**, nothing enters the knowledge graph, nothing is embedded, nothing surfaces at query time.

---

## 6. Where approved code goes

Approving a code artifact has two side effects, both idempotent:

1. **Graph edge.** An AGE graph edge is written from the existing `Algorithm` node to a new `CODE_FUNCTION` node keyed by the `code_artifact_id`. The edge type is `HAS_REFERENCE_IMPL`, stored as a property (AGE's convention) rather than a Cypher label. The node carries the file, lines, language, repo URL, and pinned commit; the edge carries the relationship.
2. **Embedding.** The snippet is embedded and upserted into a dedicated pgvector table. The embedding is keyed by `code_artifact_id`, scoped by `(tenant_id, workspace_id)`, and indexed with HNSW cosine.

Rejecting the artifact removes both sides. Re-approval is cheap (the CODE_FUNCTION node persists — only the edge and the embedding row are toggled).

### Why a dedicated embedding table (and not the workspace vector store)

EdgeQuake already has a per-workspace vector store for paper chunks. The tempting move is to tag code embeddings with `source_type: "code_function"` and reuse it.

The blocker is embedding dimension. Paper chunks typically embed at 768 dimensions (embeddinggemma); code embeds at 896 (Jina code-embeddings). pgvector fixes dimension at the column level — you cannot mix 768-d and 896-d vectors in one column. The clean fix is a separate `code_artifact_embeddings` table accessed through its own trait, keeping the "one workspace = one text embedding dimension" invariant intact.

---

## 7. The code embedder

### Model choice: Jina code-embeddings

`jina-code-embeddings-0.5b` quantised to Q8_0. It's a small (896-d) instruction-tuned embedder designed specifically for code tasks. On the Cornstack-Python retrieval benchmark it's competitive with 7B open-source alternatives while running on a single consumer GPU slot.

Why Jina over Nomic Embed Code, the other obvious pick:

| | Jina | Nomic Embed Code |
|---|---|---|
| Size | 0.5 B | 7 B |
| Dim | 896 | 3584 |
| License | Apache-2.0 | Apache-2.0 |
| Hosted on | User's llama.cpp Vulkan | Would need same treatment |
| Throughput at 1 slot | ~5× faster | — |

Smaller vectors, faster inference, one slot on the user's existing llama.cpp deployment. For Phase 1's volume (tens of approvals per paper at most), 0.5 B is more than enough accuracy.

### Hosting: llama.cpp Vulkan

`llama-server --embeddings --pooling last` on the Vulkan backend, served behind an OpenAI-compatible `/v1/embeddings` endpoint. Vulkan because the user runs AMD consumer hardware; the llama.cpp ROCm backend was less stable in testing.

Configured with 1 slot and 8 K context — bursty load at approval time, not sustained throughput. More slots would sit idle.

### Asymmetric instruction prefixes

The important subtlety: Jina's code model is **asymmetric and instruction-tuned**. Callers must prepend task-specific prefixes to *both* sides — passages (when indexing) and queries (when searching) — and the prefixes are fixed by the model. Changing them silently tanks retrieval quality with no error signal.

Specifically, for EdgeQuake's use case (natural-language query → code snippet), the `nl2code` task applies:

- Passage side (at approval time): `"Candidate code snippet:\n" + <code>`
- Query side (at query time): `"Find the most relevant code snippet given the following query:\n" + <nl_query>`

A unit test pins the exact strings so a stray refactor can't accidentally break retrieval. The passage-side prefix is identical between `nl2code` and `code2code`, so the same passage works for both query types; EdgeQuake only uses `nl2code` today.

---

## 8. Query-time enrichment

### The extra retrieval pass

A query against EdgeQuake already runs a hybrid pipeline (graph expansion + vector search over paper chunks + reranking). This feature adds a *second* retrieval pass, running in parallel to the main one:

1. After the main retrieval builds its context, collect the set of `document_id`s it surfaced.
2. Embed the user's NL query with the Jina `nl2code` query prefix.
3. HNSW cosine search over `code_artifact_embeddings`, scoped to `(tenant, workspace, document_id ∈ main_retrieval_docs, status='approved')`, capped at 5 hits by default, cosine distance ≤ 0.6.
4. Attach each hit to the context as a rendered `### Reference Code Implementations` section, and also expose them as a first-class `reference_code[]` array on the API response.

### Why scope to retrieved documents

Without document scoping, code enrichment would run workspace-wide. Consider two papers in the same workspace, both with approved implementations of "k-means clustering variants". The user asks about paper A; the main retrieval (which knows about papers) correctly returns paper A's chunks. The code enrichment (which runs a raw vector search) might return paper B's code because the cosine is marginally better. Answers would mix paper A's prose with paper B's code — accurate-looking but wrong.

The fix: push the `document_id ∈ {retrieved_docs}` filter into the HNSW search itself (into the CTE, not as a post-filter after top-K). This way the top-K is evaluated *among allowed papers*, not globally. For single-paper workspaces, behaviour is unchanged. For multi-paper, answers stay paper-consistent.

If the main retrieval surfaces no documents at all, enrichment short-circuits — no paper context means nothing to augment.

### Tunables

Two knobs, both env-overridable per deployment:

| Env | Default | What it does |
|---|---|---|
| `EDGEQUAKE_QUERY_CODE_SNIPPETS` | 5 | Hard cap on hits per query. Snippets are ~8 KB each, so 5 × 8 = 40 KB fits comfortably in any model's context. |
| `EDGEQUAKE_QUERY_CODE_MAX_DISTANCE` | 0.6 | Cosine distance ceiling (`d = 1 − cos`). Tighter thresholds suppress weak matches; looser thresholds surface more. 0.6 is permissive while pre-release ground truth is still being collected. |

### When the feature is off

The enrichment step is a pure no-op when any of:

- `EDGEQUAKE_CODE_EMBEDDING_URL` is unset (no embedder configured → feature disabled).
- Main retrieval produced no documents (nothing to enrich).
- No approved `code_artifacts` in the workspace match the query within the distance threshold.

Each condition falls through silently — the query path behaves exactly as it did before this feature, at zero extra cost. Logging surfaces *why* enrichment was skipped so operators can tell "nothing to surface" apart from "misconfigured".

---

## 9. How the LLM sees the result

Two places surface the approved code:

### 9.1 Inside the LLM prompt

The query context grew a new section that the prompt template renders verbatim:

````markdown
### Reference Code Implementations

The following code snippets are reviewer-approved implementations of
algorithms referenced above. Treat them as authoritative.

[C1] **Algorithm 1 Capacity-constrained cluster rebalancing** — `ivf_pq/rebalance.py:4-153`
     ([source](https://github.com/.../blob/18560e0/ivf_pq/rebalance.py#L4-L153))
_rationale:_ rebalance_clusters implements the capacity-constrained rebalancing algorithm by...
```python
def rebalance_clusters(...):
    ...
```
````

Two phrases pull weight. *"Reviewer-approved"* signals to the model that the snippet is not speculative — a human looked at it and confirmed the mapping. *"Treat them as authoritative"* nudges the model to cite from the code rather than paraphrase the paper's prose when the two conflict.

The GitHub link is pinned to the exact commit the reviewer approved, so the reference doesn't rot when upstream rewrites.

### 9.2 As a structured API response field

The `POST /api/v1/query` response gained a top-level `reference_code: []` array. Each entry carries the same information as the rendered block, but as structured JSON, so callers that build their own output — the OpenWebUI tool in particular — don't have to parse the LLM prompt to find the snippets.

The field is omitted when empty, preserving backwards compatibility.

---

## 10. OpenWebUI chat integration

The `edgequake_tool.py` OpenWebUI tool (v0.3.0) consumes `reference_code[]` from the query response and renders each snippet as a fenced code block in the text it returns to the OpenWebUI chat model. Parallels the existing chunks / entities / relationships sections — no special-casing at the UI level, just one more block.

Chat models wired to the tool now see approved code in their context without any further configuration. Negative queries (nothing matches within the distance threshold) produce no block at all.

---

## 11. Where the pieces live

EdgeQuake is a multi-crate Rust workspace plus some sidecar services. The feature is deliberately spread so each concern sits where its neighbours already do.

### Service-level

| Path | What it does |
|---|---|
| `code-analyzer/` | Standalone Python + FastAPI sidecar. Drives `claude` CLI headless. Owns the repo clone volume. |
| `openwebui/edgequake_tool.py` | OpenWebUI tool module. Renders API responses (including `reference_code[]`) into the chat model's context. |
| `edgequake/docker/docker-compose.yml` | Wires the `code-analyzer` service, the shared `code-analyzer-workspace` volume (read-write for analyzer, read-only for edgequake), and the `EDGEQUAKE_CODE_EMBEDDING_*` env vars pointing at the Jina llama.cpp server. |

### Rust crates

| Crate | Concern |
|---|---|
| `edgequake-pdf` | Phase 0 layer 1: extracts PDF link annotations, finds the references-section boundary. |
| `edgequake-pipeline` | Adds `CODE_REPOSITORY` to the default entity-extraction types (layer 3). |
| `edgequake-agents` | Repo detection (layer 2), code-analyzer HTTP client, snippet extractor (tree-sitter function-boundary correction), the Jina embedder client (`JinaEmbedder` + instruction prefixes), and the Postgres storage for `code_artifacts` + `code_artifact_embeddings`. |
| `edgequake-tasks` | `RepoDetection` and `CodeReferenceAnalysis` task variants. |
| `edgequake-api` | REST endpoints (`/api/v1/documents/{id}/repos`, `/api/v1/code-reference/...`), task processors wiring code-analyzer calls + snippet extraction + embedding, approval side-effects (graph edge + embedding upsert), and the `reference_code[]` response field. Wires the Jina embedder + Postgres code-vector storage at startup in `state/postgres.rs`. |
| `edgequake-storage` | Defines the `CodeVectorStorage` trait (vocabulary: `search_approved_code(tenant, workspace, query_vec, limit, max_distance, document_ids)`) and its Postgres implementation (HNSW cosine; paper-scoped variant pushes the `document_id = ANY(...)` filter into the search CTE). |
| `edgequake-query` | The post-retrieval enrichment step. Holds `Option<Arc<dyn CodeVectorStorage>>` + `Option<Arc<JinaEmbedder>>` injected via `SOTAQueryEngine::with_code_reference(...)`; both `None` = silent no-op. Knows nothing about sqlx, Postgres, or HNSW — it calls the trait. |
| `edgequake_webui` | The "Code Matches" tab, the review UI (approve / reject per candidate), React Query hooks. |

### Database migrations

- `042_add_document_repos.sql` — Phase 0 persistence of detected repos.
- `043_add_code_artifacts.sql` — `code_artifacts` + `code_reference_runs`, new task-type constraint values.
- `044_add_code_artifact_embeddings.sql` — pgvector(896) column + HNSW cosine index + tenant/workspace/algorithm/document indexes.

### Graph schema

No migration — Apache AGE is schemaless. The `CODE_FUNCTION` node type and `HAS_REFERENCE_IMPL` edge type are written on first approval. Edge type lives as a `relation_type` property (AGE's convention), not as a Cypher label.

---

## 12. Design trade-offs, summarised

Decisions that were deliberately made one way rather than another:

**Agent-driven localization vs. bespoke retriever.** Phase 1 uses Claude Code. Bespoke is cheaper per run and fully deterministic, but requires building tree-sitter + BM25 + LLM-rerank per language. Deferred to Phase 2 if cost pressure or private-repo constraints show up.

**CLI subprocess vs. Python Agent SDK.** CLI. Subscription auth is documented and supported through the CLI; SDK is API-key-only. Losing nothing by skipping the SDK — it shells out to the same binary.

**Sonnet vs. Haiku vs. Opus as default.** Sonnet. Haiku misses harder matches; Opus is overkill and draws from a constrained quota. Per-request override is still possible.

**Dedicated vector table vs. workspace-shared.** Dedicated. Code and text embedders have different dimensions; pgvector doesn't mix. A dedicated table behind a trait is cleaner than a shared-but-type-tagged one, and avoids forcing a workspace-wide embedding-dim decision.

**Trait boundary vs. inline sqlx.** Trait. An earlier draft opened a Postgres pool inside `edgequake-query` directly; that violated the crate's "no DB knowledge" invariant. The `CodeVectorStorage` trait restores the boundary — same Postgres work happens at runtime, but the query crate goes back to being storage-agnostic.

**Push document filter into HNSW CTE vs. post-filter.** Push. Post-filter would let a paper-unrelated snippet evict the right one from the top-K before the WHERE runs. Pushing the filter into the CTE keeps top-K honest.

**Jina vs. Nomic Embed Code.** Jina. Smaller vectors (896 vs 3584), ~5× faster on the user's consumer GPU, competitive quality at this scale. Nomic wins at >7B-scale workloads; this isn't one.

**Auto-approve vs. human review.** Human review throughout. Both repo selection and individual code-artifact approval require explicit action. Mirrors the existing algorithm / entity review posture, which users already trust.

**Detect repo during ingestion vs. lazily on first query.** Ingestion. Users don't discover they need a repo until they ask a code question — lazy detection would block the first query while clone + localize runs. Eagerly detecting means the repo panel is already populated by the time the user wonders about it.

---

## 13. Tools and services used

External dependencies introduced by this feature:

| Tool | Purpose | Why chosen |
|---|---|---|
| **Claude Code CLI** (`@anthropic-ai/claude-code`) | Headless code-localization agent | Read-only tool allow-list matches the task; subscription auth documented; structured JSON output. |
| **Jina code-embeddings 0.5B** (Q8_0 GGUF) | Code embedding for approval-time indexing and query-time search | Small, instruction-tuned for code, strong on NL→code retrieval, runs on one consumer GPU slot. |
| **llama.cpp** (Vulkan backend) | Serves the Jina embedder via OpenAI-compatible HTTP | Already deployed on the user's host. Vulkan avoids ROCm instability on consumer AMD hardware. |
| **pgvector** + HNSW | Dedicated cosine-similarity index for code embeddings | Same extension EdgeQuake already uses for paper chunks; HNSW handles the expected O(10³) snippets per workspace without tuning. |
| **Apache AGE** | Storing the `Algorithm ──HAS_REFERENCE_IMPL──▶ CODE_FUNCTION` edge | Already EdgeQuake's graph backend; schemaless, so no migration. |
| **tree-sitter** | Function-boundary correction when the LLM returns approximate line ranges | Language-aware, incremental, battle-tested. Per-language grammars loaded on demand. |
| **pdfium-render** | Extracting PDF link annotations for Phase 0 layer A | Already a transitive dep (via `edgequake-pdf2md`), so no new dependency. |
| **git2 (libgit2)** | Shallow, blobless repo clone in code-analyzer | Battle-tested, well-supported. `gix` shallow support is newer; revisit post-1.0. |
| **SearXNG** + **Crawl4AI** | Phase 0 layer B (web-search fallback) | Both self-hosted by the user; env-gated so deployments without them degrade gracefully to "layer B off". |
| **FastAPI** + **GitPython** | HTTP surface + clone driver for the code-analyzer sidecar | Minimal, widely known, small surface area. |

---

## 14. Phase 2 — Reference Codebase RAG (shipped)

A separate codebase RAG index for approved reference repos, targeted at coding-agent workflows — porting a C/Python reference implementation into Rust, integrating an attention kernel, validating a paper's pseudocode against its actual implementation.

Phase 2's design deliberately mirrors Phase 1's shape: trait behind a Postgres adapter, Jina code embedder reused on the passage + query side, and per-tenant/workspace isolation throughout. What's new is a **symbol graph** built alongside the vector index, so the agent can walk callers/callees instead of purely fishing by NL similarity.

### 14.1 What happens when you approve a code match

Once `code_artifact.status` flips to `approved`, three side-effects fire in the review handler (`crates/edgequake-api/src/handlers/code_reference/mod.rs`):

1. **Graph edge.** Algorithm → CODE_FUNCTION in AGE, per Phase 1.
2. **Jina embedding.** Snippet → `code_artifact_embeddings`, per Phase 1.
3. **Auto-enqueue** (new). If `EDGEQUAKE_REFERENCE_CODEBASE_AUTO_INDEX=true`, a `reference_codebase_index` task is enqueued. Race-safe: `INSERT … ON CONFLICT DO NOTHING` against the `(tenant, workspace, document_repo, commit, mode)` uniqueness key, so sibling approvals on the same repo enqueue exactly one indexing task. Fire-and-forget (`tokio::spawn`) — the approval HTTP response doesn't wait on the clone.

Auto-triggered rows carry `auto_triggered=TRUE` and a reduced `max_files_override` (default 5000) to prevent one click burning the Jina quota on a 20k-file repo. Explicit `POST /indexes` keeps the full cap.

### 14.2 The indexer

Runs as the `reference_codebase_index` task:

1. **Snapshot.** `code-analyzer /snapshot` clones or reuses the repo on the shared `code-analyzer-workspace` volume. EdgeQuake never clones — the analyzer owns the disk and edgequake mounts read-only.
2. **Scan.** `.gitignore`-aware walk of the repo root, skipping `.git / target / node_modules / dist / build / .venv / vendor`. Honours `max_files`, `max_file_bytes`, `max_chunks` caps.
3. **Symbols + edges.** Tree-sitter via `ast-grep-core` for Rust / Python / TypeScript / C / C++; regex fallback for everything else. Per-language edge coverage:

| Edge | Rust | Python | TS | C | C++ |
|---|---|---|---|---|---|
| `defines` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `calls` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `imports` | `use` | `import` | `import`/`require` | `#include` | `#include` |
| `references` | ✓ | ✓ | ✓ | — | — |
| `implements` | `impl Trait for Type` | — | `implements` | — | deferred (Phase 3 / SCIP) |
| `inherits` | — | `class X(Y)` | `extends` | — | deferred (Phase 3 / SCIP) |

C's preprocessor macros are **not expanded** — macro-hidden calls don't resolve. Accepts the approximation; SCIP is the Phase-3 precision pass.

4. **Chunker.** In `algorithm_focused` mode, keeps only symbols inside files that contain an approved `code_artifact` anchor (or that *are* anchors). Each chunk carries `algorithm_focus` ∈ [0,1] — anchors get 1.0, file-neighbours get 0.35. `full` mode chunks every symbol.
5. **Embed.** Each chunk goes through `JinaEmbedder.embed_code_for_indexing` (same passage prefix as Phase 1, so Phase 1 + Phase 2 embeddings live in a compatible space). Upserted into `reference_codebase_embeddings` keyed by `chunk_id`.
6. **Complete.** Status advances `queued → scanning → parsing → chunking → embedding → complete`. Any grammar crash bumps the per-file `parse_errors` counter without failing the whole index.

### 14.3 Data model

Five new tables (migrations 045, 046, 048):

| Table | Role |
|---|---|
| `reference_codebase_indexes` | Per-commit build status, counts, timings, `auto_triggered`, `max_files_override` |
| `reference_codebase_files` | Scanned source files + `parse_errors` telemetry |
| `reference_codebase_symbols` | Functions, structs, classes, impls with line spans |
| `reference_codebase_edges` | `defines / calls / imports / references / implements / inherits` edges |
| `reference_codebase_chunks` | Retrieval units (`algorithm_anchor / symbol / file_overview`) with `algorithm_focus` |
| `reference_codebase_embeddings` | pgvector(896) + HNSW cosine on chunk embeddings |

Graph edges live in the SQL tables, not AGE — the edge count per index is easily 5k+ (the V3DB zk-ivf-pq index is 3959), which would outpace AGE's Cypher-query tooling. The subgraph endpoint runs a recursive CTE directly against the SQL tables, bounded by `hops` and `max_nodes`.

### 14.4 Retrieval APIs

Four endpoints under `/api/v1/reference-codebase/*`:

- `POST /indexes` — enqueue (manual path; auto-enqueue uses the same pipeline).
- `GET /indexes/{id}` — status, counts, errors.
- `GET /by-repo/{document_repo_id}` — list every index for a repo (newest first).
- `GET /indexes/{id}/graph?anchor_artifact_id=…|anchor_symbol=…&hops=1&max_nodes=200` — BFS subgraph for the Code Graph tab and the `get_symbol_neighborhood` MCP tool. Recursive CTE over `reference_codebase_edges`; `hops` clamps to [1,3], `max_nodes` to [1,1000]. Rejects when the index isn't `status='complete'`. Seed resolution: artifact id → overlapping symbols, symbol name → exact-then-case-insensitive, neither → top-N by degree (fallback for `mode='full'`).
- `POST /query` — semantic search via Jina. `document_repo_id` / `index_id` / `algorithm_ids` filters stack. Anchor-focus boost applied in SQL (`algorithm_focus * -0.08 + cosine_distance`).

### 14.5 MCP tools

Two new tools in the Node.js MCP server:

- **`query_code(query, repo_url?, algorithm_id?, limit?, max_distance?)`** — semantic NL → code-chunk search. Returns rendered markdown with fenced code blocks, file:line ranges, and GitHub deep-links pinned to `repo_commit`. Meant for "find the attention kernel" / "how does the rebalance loop work" agent prompts.
- **`get_symbol_neighborhood(index_id, symbol_name?, anchor_artifact_id?, hops?, max_nodes?)`** — N-hop symbol walk. Returns nodes + outgoing edges grouped per source symbol, with anchor markers (⭐) for approved artifacts. Meant for "what calls this" / "what does this depend on" follow-ups.

Both tools share the existing env-bound tenant/workspace singleton — caller tools do not accept tenant args (multi-tenant leak guard).

### 14.6 Code Graph WebUI tab

New tab on the document detail page next to Code Matches. Left pane lists approved code artifacts (anchors); clicking one fetches the N-hop subgraph and renders it with sigma.js. Node click opens a drawer with the chunk's file + line range and a pinned-commit GitHub link. Truncation hints and in-flight index status banners keep the UI honest about partial results. Runs on the same graphology + ForceAtlas2 pipeline the paper knowledge graph uses; code-specific colouring (hue per language, anchors brighter) lives in a dedicated `CodeGraphRenderer` so the entity renderer stays clean.

### 14.7 Where it shares with Phase 1

- **Embedder.** `JinaEmbedder` is reused verbatim; Phase 1 passage prefix + Phase 2 chunk passage prefix are both `Nl2Code::passage_prefix` so a chunk embedding of `fn rebalance_clusters` is directly comparable to a Phase 1 approval of the same function.
- **Graph edge.** Phase 1 writes one AGE edge per approved artifact; Phase 2 writes N SQL edges per indexed symbol. The AGE edge stays — it drives Phase 1 query enrichment. The SQL edges are purely for Phase 2's coding-agent retrieval.
- **Review posture.** Nothing in Phase 2 triggers from unapproved state — a repo must be approved (Phase 0), at least one code match must be approved (Phase 1), and then either auto-enqueue or an explicit POST kicks Phase 2 indexing.

---

## 15. Not yet shipped — Phase 3 and beyond

- **SCIP integration.** Env flag `EDGEQUAKE_REFERENCE_CODEBASE_SCIP=off|auto|required` is declared but unwired. Would unlock accurate `implements` / `inherits` for C++ (templates, multiple inheritance) and type-aware `references` for C / C++. Requires per-language indexers (gopls, pyright, scip-rust, scip-cpp).
- **Agent-driven hard-case localization.** A ReAct-style agent that goes deeper on tricky matches. Phase 3 if the deterministic CLI flow proves inadequate on specific paper classes.
- **Multi-repo per paper.** Some papers split "model" and "dataset" repos. Phase 1 + 2 handle one repo per paper; loop later.
- **Automatic re-analysis on upstream commits.** Explicit user action only for now — avoids surprise edits to the knowledge base when upstream rewrites.
- **Cross-paper repo dedup.** Same repo cited by Paper A and Paper B produces two indexes. Correct but wasteful; a `(repo_url, repo_commit)` dedup layer is a follow-up.
- **Git-blame-style provenance per snippet line.** Nice-to-have for attribution.
- **Approved-algorithm vector retrieval.** Algorithms *are* embedded at approval time (into the workspace vector store with `type: "algorithm"` metadata), but the SOTA retrieval filter currently only surfaces `{chunk, entity, relationship}` vectors — the algorithm vectors exist but aren't reachable via `/query` today. Follow-up: add `AlgorithmVectorStorage` symmetric to `CodeVectorStorage`, surface as a parallel `approved_algorithms[]` response field.
- **Clone storage GC.** Phase 2 persists clones on the shared volume; nothing trims them. A TTL sweeper (`EDGEQUAKE_REFERENCE_CODEBASE_CLONE_TTL_DAYS`) is documented but not yet implemented.

---

## 16. Operational notes

**Health.** Three services participate:
- `edgequake` — check `GET /health`; confirms schema migration version and wired providers.
- `code-analyzer` — check `GET :9100/health`; returns the `claude` CLI version when the subscription token is valid.
- Jina embedder — check `GET :11437/health`; the OpenAI-compatible endpoint responds `{"status":"ok"}` when the server is loaded.

**Token rotation.** `CLAUDE_CODE_OAUTH_TOKEN` is valid for one year from issue. A 401 in the analyzer container means it's time to regenerate via `claude setup-token` on the host and redeploy.

**Quota behaviour.** Analyzer calls share the subscription's 5-hour rolling and weekly quotas with interactive Claude Code use. A 429 from the analyzer bubbles up as a task failure with a clear message — the edgequake task processor doesn't auto-retry across quota resets; the user re-triggers after the 5h window.

**Rollout safety.** The feature gates itself off cleanly when:
- The Jina embedder URL is unset (approval-time embedding becomes a no-op; query-time enrichment becomes a no-op).
- `CLAUDE_CODE_OAUTH_TOKEN` is unset (code-analyzer refuses to run, but the rest of EdgeQuake is unaffected).
- `SEARXNG_URL` is unset (Phase 0 layer B disabled; layers A and C still work).

None of these states produce loud errors — each individual capability degrades to "feature off".
