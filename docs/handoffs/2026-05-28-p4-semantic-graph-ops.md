# P4 — Semantic Operations on the Graph

**Date:** 2026-05-28
**Origin:** Conversation in `~/repos/lattice` (downstream consumer). No
companion plan/PR exists yet — this handoff is the only artifact.
**Audience:** Whoever picks up edgequake-side work to support
hybrid-graph+vector operations and richer graph viz primitives.

---

## Why this is in the edgequake repo

Lattice is a downstream consumer of edgequake's REST + graph API. While
auditing lattice's KG UX we identified four pain points (P1–P4); three
of them are best fixed in **edgequake** because they need new server-side
endpoints, precomputed data on `nodes`, or Cypher/SQL that crosses AGE +
pgvector. This handoff is scoped to **P4 only** — the broader set of
semantic operations on the graph. P3 (subgraph-from-query) is the
sibling feature; the next session will plan it separately
(see "Next session" below).

The lattice repo holds the pinned types in
`/home/timo/repos/lattice/src/edgequake/edgequake.types.ts` (Phase-2
graph API surface as of 2026-05-27) and is the source of the user-facing
complaints. Lattice's entity extractor uses Claude Haiku 4.5 with
workspace-defined `entity_types[]` plus Plan-3 AST structural entities
(`Heading`, `Table`, `CodeBlock`, `Quote`, `Citation`), so the graph is
property-graph-ish with typed nodes/edges and document-scoped chunk
provenance.

## The structural advantage to exploit

Edgequake stores **both** the property graph (Apache AGE) and embeddings
(pgvector) in the **same Postgres instance**. This means a hybrid
predicate like `WHERE vec_distance(n.embedding, $q) < $t` inside a
Cypher traversal is achievable in a single round-trip. Neo4j needs the
vector-index plugin to do this; you already have it as a structural
property of the schema. P4 is fundamentally about *surfacing* that
capability through the API.

---

## User stories

| ID    | Story |
|-------|-------|
| US-4.1 | **Hybrid query.** "Among entities of type `Paper` within 2 hops of `Graph Attention Networks`, rank by semantic similarity to `'attention as message passing'` and show top 20." Graph filter + vector rerank in one pass. |
| US-4.2 | **Semantic bridge.** "What concepts connect document A and document B?" — shortest paths between A's entities and B's entities, weighted by edge type + embedding distance. |
| US-4.3 | **Cluster and summarise.** "Run Leiden on my workspace, label each community with a Haiku-generated summary, let me hover a cluster for the summary and double-click to expand." |
| US-4.4 | **Structural + semantic filter.** "Show high-betweenness entities semantically close to query Q." Centrality combined with similarity — finds *connector* concepts. |
| US-4.5 | **Similarity pruning.** On a current canvas, hide nodes whose embedding is >`τ` cosine distance from the focal node. Pure noise filter, no expansion. |
| US-4.6 | **"Why is this here?" explanation.** For any node, show the traversal path that brought it in (which query, which hops, which edge types). |

## Functional requirements

| ID    | Requirement | Where it lives |
|-------|-------------|----------------|
| FR-4.1 | Cypher / REST supports `vec_distance(n.embedding, $q) < $t` predicates inside traversal | `crates/edgequake-api/src/handlers/graph/` + AGE Cypher function or REST shim that composes the SQL |
| FR-4.2 | Server-side Leiden community detection precomputed on sync; `community_id` + `community_summary` columns per node | New sync hook OR `POST /api/v1/graph/communities/recompute`. Lattice supplies the summary text via Haiku and writes through `PUT /graph/entities/{id}/body`. |
| FR-4.3 | Path-finding endpoint: `shortest_paths(source_set[], target_set[], edge_types[], max_hops, weight ∈ {uniform, degree, embedding_distance})` | New: `POST /api/v1/graph/paths` |
| FR-4.4 | Centrality / PageRank precomputed per node or computed on a passed subgraph | Server-side preferred; AGE has `pagerank` via SQL pipe |
| FR-4.5 | Per-node traversal provenance returned with every subgraph response (`{origin_query?, hops, via_edge_types}`) | Extends existing `GET /api/v1/graph` and the new `/subgraph` endpoint |
| FR-4.6 | UI exposes these as an "operations palette" alongside the filter rail, not buried in Cypher | `edgequake_webui/src/components/graph/` |

## Minimum new API surface

```
POST  /api/v1/graph/subgraph
  { start_nodes[], depth, edge_types[], entity_types[],
    document_ids[], max_nodes,
    semantic_filter?: { query_text, max_cosine_distance } }

POST  /api/v1/graph/paths
  { sources[], targets[], max_hops, edge_types[],
    weight: "uniform" | "degree" | "embedding_distance" }

POST  /api/v1/graph/communities/recompute
  { algorithm: "leiden" | "louvain", resolution? }
  -> writes community_id per node, returns community list

POST  /api/v1/graph/query
  # Cypher escape hatch that accepts vec_distance() in predicates
  { cypher, params }
```

Plus `nodes` schema additions: `community_id INT`, `community_summary
TEXT`, `pagerank REAL` (nullable, populated by FR-4.4).

## Tool reference matrix (for inspiration / prior art)

| Tool | P4 coverage | What to steal |
|------|-------------|---------------|
| Neo4j 5.18+ (Cypher + vector index) | Full (mechanism) | `db.index.vector.queryNodes()` inline with `MATCH` traversal — the canonical hybrid pattern. We can do the equivalent with AGE Cypher + a custom `vec_distance` function. |
| Memgraph + MAGE + GraphChat | Full | MAGE ships PageRank/Louvain/community-detection as Cypher procs. Worth replicating the proc-style API surface. |
| Microsoft GraphRAG | Full (opinionated) | Leiden + per-community LLM summaries + global/local search modes. **Closest match for FR-4.2 + FR-4.3.** |
| Graphistry + GFQL + MCP server | Full | GFQL = Cypher-subset + dataframe ops + vector ops. MCP server exposes them to agents. Worth studying for the "operations palette" concept. |
| Tom Sawyer Perspectives | Full (commercial) | Pattern-query UX. |
| yFiles / KeyLines / ReGraph | Rendering only | Combos / LOD / edge bundling — UI patterns to bring into the WebUI later. |
| Current edgequake WebUI (Sigma.js + graphology) | None of P4 today | `graphology-communities-louvain` is already in deps but unused. |
| AGE Viewer | None | Cypher-only, no vector. |

## What lattice will do once this lands

- Replace its current single-document graph view with starting-set
  subgraph rendering (P3, planned separately).
- Run Leiden on sync (`lattice sync --compute-communities`), write
  `community_id` and Haiku-generated `community_summary` via PUT
  through the new write paths.
- Surface FR-4.1 hybrid queries in the lattice CLI (`lattice rels` and
  `lattice search`) before/while the WebUI palette is built.

## Out of scope here

- **P1** (multi-document filter on `/api/v1/graph`) — separate, smaller
  change. Wire `MetadataFilter.document_ids` (already in storage layer,
  `crates/edgequake-storage/src/traits/vector.rs`) into the graph
  handler.
- **P2** (visual overload) — primarily a WebUI concern: LOD,
  semantic zoom, combos. Communities (FR-4.2) cover the data side.
- **P3** (subgraph from semantic query) — the *next* planning session.
  Treat P3 as the primary on-canvas entry point; P4 endpoints are the
  building blocks P3 will compose. Plan-of-record: Neo4j LLM Knowledge
  Graph Builder is the prior-art exemplar.

## Pointers

- Lattice pinned types (what the consumer expects today):
  `/home/timo/repos/lattice/src/edgequake/edgequake.types.ts`
- Current graph handler:
  `crates/edgequake-api/src/handlers/graph/`
- Current WebUI graph viewer:
  `edgequake_webui/src/components/graph/` (Sigma.js + graphology, MIT)
- Storage trait with already-defined `MetadataFilter`:
  `crates/edgequake-storage/src/traits/vector.rs`
- Open upstream specs that touch this area:
  `specifications/0003_explainability_issue_128/` (provenance, OPEN),
  `specs/005-filter.md` (document filters, query-only today)

## Next session

Recommended invocation in the **lattice** repo (the consumer
perspective shapes the requirements, but the implementation will land
here in edgequake):

```
/grill-me Plan P3 — subgraph from semantic query, modelled on
Neo4j LLM Knowledge Graph Builder; primarily an edgequake-side
feature that lattice benefits from.
```

The grill should pressure-test:
- whether P3 needs anything beyond the `/api/v1/graph/subgraph`
  endpoint sketched above
- how chunk-level vs entity-level provenance is rendered (LLM KG
  Builder shows both — lexical graph and entity graph toggleable)
- the exact join from vector hit → starting set (chunk-hit ⇒
  containing-entities? top-K entities directly via entity
  embeddings? both?)
- whether comparison view (two starting sets, color-coded) belongs
  in v1 or v2

When the P3 plan is ready, the two handoffs together cover the full
edgequake-side roadmap to unblock lattice's KG UX.
