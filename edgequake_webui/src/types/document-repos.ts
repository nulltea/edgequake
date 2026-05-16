/**
 * Reference-repository detection types for the frontend.
 *
 * Mirrors the REST DTOs in `crates/edgequake-api/src/handlers/repos/types.rs`
 * plus the domain types in `crates/edgequake-agents/src/repo_detection/types.rs`.
 * Keep these in sync if the backend shape changes.
 */

export type RepoHost = "github" | "gitlab" | "bitbucket";

export type DetectionMethod = "pdf_link" | "github_api" | "web_search" | "manual";

export type RepoConfidence = "high" | "medium" | "low";

export type RepoStatus = "pending" | "approved" | "rejected";

export type DetectionRunStatus = "running" | "complete" | "failed";

/**
 * Post-detection verifier verdict. Set by the LLM verifier that runs after
 * Layer A or Layer B produces a candidate. All four fields are absent when
 * the verifier hasn't run (old rows, or a run where the verifier was
 * disabled / failed). Surfaced purely as an advisory signal in the UI —
 * reviewers still decide approve/reject.
 */
export type VerificationVerdict =
  | "official"
  | "third_party"
  | "unrelated"
  | "inconclusive";

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
  /** Omitted when the verifier hasn't run for this row. */
  verification_verdict?: VerificationVerdict;
  /** 0..1 confidence reported by the verifier LLM. */
  verification_confidence?: number;
  /** One-sentence rationale from the verifier. */
  verification_rationale?: string;
  /** Timestamp when the verifier last ran for this row. */
  verified_at?: string;
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
