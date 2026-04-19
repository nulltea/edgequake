/**
 * Reference-repository detection types for the frontend.
 *
 * Mirrors the REST DTOs in `crates/edgequake-api/src/handlers/repos/types.rs`
 * plus the domain types in `crates/edgequake-agents/src/repo_detection/types.rs`.
 * Keep these in sync if the backend shape changes.
 */

export type RepoHost = "github" | "gitlab" | "bitbucket";

export type DetectionMethod = "pdf_link" | "web_search";

export type RepoConfidence = "high" | "medium" | "low";

export type RepoStatus = "pending" | "approved" | "rejected";

export type DetectionRunStatus = "running" | "complete" | "failed";

export interface RepoCandidate {
  id: string;
  document_id: string;
  host: RepoHost;
  owner: string;
  repo: string;
  url: string;
  detection_method: DetectionMethod;
  pdf_page_index: number | null;
  search_rank: number | null;
  source_url: string | null;
  confidence: RepoConfidence;
  status: RepoStatus;
  created_at: string;
  updated_at: string;
}

export interface DetectionRun {
  document_id: string;
  status: DetectionRunStatus;
  layer_a_candidates: number;
  layer_b_candidates: number;
  error_message: string | null;
  attempted_at: string;
  completed_at: string | null;
}

/** Response from GET /api/v1/repos/by-document/{document_id}. */
export interface RepoListResponse {
  document_id: string;
  candidates: RepoCandidate[];
  detection_run: DetectionRun | null;
}

/** Response from POST /api/v1/repos/{repo_id}/review. */
export interface RepoReviewResponse {
  id: string;
  status: RepoStatus;
}

/** Response from POST /api/v1/repos/detect/{document_id}. */
export interface DetectReposResponse {
  document_id: string;
  track_id: string;
  status: string;
}
