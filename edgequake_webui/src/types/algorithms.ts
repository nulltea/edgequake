/**
 * Algorithm extraction types for the frontend.
 *
 * Mirrors the SDK types for use in the WebUI.
 */

/** A single step in an algorithm. */
export interface AlgorithmStep {
  number: number;
  action: string;
  details: string;
  math?: string;
}

/** An input or output of an algorithm. */
export interface AlgorithmIO {
  name: string;
  type: string;
  description: string;
}

/** Full algorithm entity. */
export interface Algorithm {
  id: string;
  tenant_id: string;
  workspace_id: string;
  document_id: string;
  name: string;
  /** Kind of construct: "Algorithm" | "Protocol" | "Functionality" | "Theorem" | "Definition" | "Lemma" | "Scheme" | custom. */
  algorithm_type?: string;
  description?: string;
  steps: AlgorithmStep[];
  inputs: AlgorithmIO[];
  outputs: AlgorithmIO[];
  preconditions: string[];
  complexity?: string;
  mathematical_notation?: string;
  pseudocode?: string;
  tags: string[];
  confidence: string;
  status: "pending" | "approved" | "rejected";
  verification_status?: string;
  verification_details?: unknown;
  created_at: string;
  updated_at: string;
}

/** Response from POST /algorithms/extract. */
export interface AlgorithmExtractionResponse {
  document_id: string;
  status: string;
  message: string;
}

/** Response from GET /algorithms/by-document/{document_id}. */
export interface AlgorithmListResponse {
  algorithms: Algorithm[];
  total: number;
}

/** Response from GET /algorithms/search. */
export interface AlgorithmSearchResponse {
  algorithms: Algorithm[];
  total: number;
  limit: number;
  offset: number;
}

/** Response from POST /algorithms/{id}/review. */
export interface AlgorithmReviewResponse {
  id: string;
  status: string;
}

/** Response from POST /algorithms/by-document/{document_id}/submit. */
export interface AlgorithmSubmitResponse {
  document_id: string;
  approved_count: number;
  rejected_count: number;
  status: string;
}

/** Response from DELETE /algorithms/by-document/{document_id}. */
export interface AlgorithmDeleteResponse {
  deleted: number;
}
