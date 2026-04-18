/**
 * Reference-code analysis types for the frontend.
 *
 * Mirrors REST DTOs in `crates/edgequake-api/src/handlers/code_reference/types.rs`
 * plus the domain types in `crates/edgequake-agents/src/code_analysis/types.rs`.
 */

export type ArtifactStatus = "pending" | "approved" | "rejected";
export type MatchConfidence = "high" | "medium" | "low";
export type CodeRunStatus =
  | "queued"
  | "cloning"
  | "analyzing"
  | "embedding"
  | "awaiting_review"
  | "complete"
  | "failed";

export interface CodeArtifact {
  id: string;
  document_id: string;
  algorithm_id: string;
  document_repo_id: string;
  repo_commit: string;
  repo_license: string | null;
  language: string;
  file_path: string;
  symbol_name: string | null;
  start_line: number;
  end_line: number;
  snippet: string;
  match_rationale: string | null;
  match_confidence: MatchConfidence;
  status: ArtifactStatus;
  created_at: string;
  updated_at: string;
}

export interface CodeReferenceRun {
  document_id: string;
  document_repo_id: string;
  status: CodeRunStatus;
  algorithm_count: number;
  finding_count: number;
  /** Theoretical API-equivalent dollar cost. On a subscription this is a
   * quota gauge, not a real charge. */
  cost_usd_equivalent: number | null;
  error_message: string | null;
  attempted_at: string;
  completed_at: string | null;
}

/** Response from GET /api/v1/code-reference/by-document/{doc_id}. */
export interface CodeReferenceListResponse {
  document_id: string;
  candidates: CodeArtifact[];
  runs: CodeReferenceRun[];
}

/** Response from POST /api/v1/code-reference/{id}/review. */
export interface CodeArtifactReviewResponse {
  id: string;
  status: ArtifactStatus;
}

/** Response from POST /api/v1/code-reference/analyze/{document_repo_id}. */
export interface AnalyzeCodeReferenceResponse {
  document_id: string;
  document_repo_id: string;
  track_id: string;
  status: string;
}
