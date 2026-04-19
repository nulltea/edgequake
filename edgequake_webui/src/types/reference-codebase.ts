/**
 * Reference codebase RAG types.
 *
 * Mirrors `/api/v1/reference-codebase/*` DTOs.
 */

export type ReferenceCodebaseIndexMode = "algorithm_focused" | "full";
export type ReferenceCodebaseIndexStatus =
  | "queued"
  | "scanning"
  | "parsing"
  | "chunking"
  | "embedding"
  | "complete"
  | "failed";

export interface CreateReferenceCodebaseIndexRequest {
  document_repo_id: string;
  mode?: ReferenceCodebaseIndexMode;
  force_reindex?: boolean;
}

export interface CreateReferenceCodebaseIndexResponse {
  document_repo_id: string;
  document_id: string;
  track_id: string;
  status: string;
}

export interface ReferenceCodebaseIndex {
  id: string;
  document_id: string;
  document_repo_id: string;
  repo_url: string;
  repo_commit: string;
  repo_path: string;
  repo_license: string | null;
  mode: ReferenceCodebaseIndexMode;
  status: ReferenceCodebaseIndexStatus;
  language_set: string[];
  file_count: number;
  symbol_count: number;
  chunk_count: number;
  edge_count: number;
  /**
   * Approximate graph diameter — the max BFS depth from the
   * highest-degree symbol. Drives the Code Graph tab's hops slider
   * upper bound. Omitted for indexes that aren't `complete` yet.
   */
  max_depth?: number | null;
  error_message: string | null;
}

export interface ReferenceCodebaseQueryRequest {
  query: string;
  document_repo_id?: string;
  index_id?: string;
  algorithm_ids?: string[];
  limit?: number;
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

// ── Subgraph (Code Graph tab) ──────────────────────────────────

export interface ReferenceCodebaseGraphNode {
  symbol_id: string;
  name: string;
  qualified_name: string;
  kind: string;
  language: string;
  file_path: string;
  start_line: number;
  end_line: number;
  /** BFS distance from the seed set (seeds=0). */
  depth: number;
  chunk_id: string | null;
  /** True when the symbol overlaps an approved code_artifact. */
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
  mode: ReferenceCodebaseIndexMode;
  hops: number;
  truncated: boolean;
  seed_symbol_ids: string[];
  nodes: ReferenceCodebaseGraphNode[];
  edges: ReferenceCodebaseGraphEdge[];
}
