/**
 * Reference-codebase types — Phase 2 of the Reference Code GraphRAG
 * extension. These mirror the Rust DTOs in
 * `edgequake/crates/edgequake-api/src/handlers/reference_codebase/mod.rs`.
 *
 * Two surfaces this SDK wraps:
 *   1. Semantic search over approved code chunks (POST /query).
 *   2. Anchor-centric symbol-graph subgraphs (GET /indexes/{id}/graph).
 *
 * @module types/reference-codebase
 */

// ── Index lifecycle ─────────────────────────────────────────────

export interface CreateReferenceCodebaseIndexRequest {
  document_repo_id: string;
  mode?: "algorithm_focused" | "full";
  force_reindex?: boolean;
}

export interface CreateReferenceCodebaseIndexResponse {
  document_repo_id: string;
  document_id: string;
  track_id: string;
  status: "queued";
}

export interface ReferenceCodebaseIndex {
  id: string;
  document_id: string;
  document_repo_id: string;
  repo_url: string;
  repo_commit: string;
  repo_path: string;
  repo_license: string | null;
  mode: "algorithm_focused" | "full";
  status:
    | "queued"
    | "scanning"
    | "parsing"
    | "chunking"
    | "embedding"
    | "complete"
    | "failed";
  language_set: string[];
  file_count: number;
  symbol_count: number;
  chunk_count: number;
  edge_count: number;
  error_message: string | null;
}

// ── Semantic search ─────────────────────────────────────────────

export interface ReferenceCodebaseQueryRequest {
  /** Natural-language query, embedded via Jina `nl2code` query prefix. */
  query: string;
  /** Restrict to a specific paper's repo. */
  document_repo_id?: string;
  /** Restrict to a specific index row (tighter than repo filter). */
  index_id?: string;
  /**
   * Restrict to chunks whose anchor (approved `code_artifact`) is tied
   * to one of these algorithm ids. Useful when the agent already knows
   * the algorithm it wants.
   */
  algorithm_ids?: string[];
  /** Max hits (default 12). */
  limit?: number;
  /** Cosine-distance ceiling (default 0.65). Smaller = stricter. */
  max_distance?: number;
}

export interface CodingContextHit {
  chunk_id: string;
  index_id: string;
  document_id: string;
  document_repo_id: string;
  repo_url: string;
  repo_commit: string;
  file_path: string;
  language: string;
  symbol_name: string | null;
  start_line: number;
  end_line: number;
  chunk_kind: string;
  algorithm_id: string | null;
  content: string;
  cosine_distance: number;
}

export interface ReferenceCodebaseQueryResponse {
  coding_context: CodingContextHit[];
}

// ── Subgraph around an anchor ──────────────────────────────────

export interface ReferenceCodebaseGraphParams {
  /** Approved `code_artifact` — resolves to overlapping symbol(s). */
  anchor_artifact_id?: string;
  /** Exact (or case-insensitive) symbol-name lookup. */
  anchor_symbol?: string;
  /** BFS depth. 1 by default, 3 max. */
  hops?: number;
  /** Hard cap on node count in the response. 200 default. */
  max_nodes?: number;
}

export interface ReferenceCodebaseGraphNode {
  symbol_id: string;
  name: string;
  qualified_name: string;
  kind: string;
  language: string;
  file_path: string;
  start_line: number;
  end_line: number;
  /** BFS distance from the seed set. Seeds are depth 0. */
  depth: number;
  /**
   * Best-matching chunk id for one-hop click-to-code in a UI.
   * `algorithm_anchor` chunks win over plain `symbol` chunks.
   */
  chunk_id: string | null;
  /** True when the symbol overlaps an approved `code_artifact`. */
  is_anchor: boolean;
  algorithm_focus: number;
}

export interface ReferenceCodebaseGraphEdge {
  source_symbol_id: string;
  target_symbol_id: string;
  kind: string;
  target_name: string | null;
}

export interface ReferenceCodebaseGraphResponse {
  index_id: string;
  document_id: string;
  repo_url: string;
  repo_commit: string;
  mode: "algorithm_focused" | "full";
  hops: number;
  truncated: boolean;
  seed_symbol_ids: string[];
  nodes: ReferenceCodebaseGraphNode[];
  edges: ReferenceCodebaseGraphEdge[];
}
