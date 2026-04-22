/**
 * Algorithms resource — extract, list, search, review, and delete algorithms.
 *
 * @module resources/algorithms
 * @see edgequake/crates/edgequake-api/src/handlers/algorithms.rs
 */

import type {
  AlgorithmDeleteResponse,
  AlgorithmExtractionResponse,
  AlgorithmListResponse,
  AlgorithmReviewResponse,
  AlgorithmSearchResponse,
  AlgorithmSubmitResponse,
} from "../types/algorithms.js";
import { Resource } from "./base.js";

/** Search parameters for the algorithms search endpoint. */
export interface AlgorithmSearchParams {
  query?: string;
  document_id?: string;
  limit?: number;
  offset?: number;
}

/** Algorithms resource for algorithm extraction and management. */
export class AlgorithmsResource extends Resource {
  /**
   * Extract algorithms from a document.
   *
   * Triggers the algorithm extraction pipeline for the given document.
   */
  async extract(documentId: string): Promise<AlgorithmExtractionResponse> {
    return this._post("/api/v1/algorithms/extract", {
      document_id: documentId,
    });
  }

  /**
   * List algorithms extracted from a specific document.
   *
   * @param documentId - The document to list algorithms for.
   * @param status - Optional status filter (pending, approved, rejected).
   */
  async listByDocument(
    documentId: string,
    status?: string,
  ): Promise<AlgorithmListResponse> {
    const params = new URLSearchParams();
    if (status) params.set("status", status);
    const qs = params.toString();
    return this._get(
      `/api/v1/algorithms/by-document/${documentId}${qs ? `?${qs}` : ""}`,
    );
  }

  /**
   * Search algorithms across the workspace.
   *
   * @param params - Search parameters (query, document_id, limit, offset).
   */
  async search(
    params?: AlgorithmSearchParams,
  ): Promise<AlgorithmSearchResponse> {
    const sp = new URLSearchParams();
    if (params?.query) sp.set("query", params.query);
    if (params?.document_id) sp.set("document_id", params.document_id);
    if (params?.limit != null) sp.set("limit", String(params.limit));
    if (params?.offset != null) sp.set("offset", String(params.offset));
    const qs = sp.toString();
    return this._get(`/api/v1/algorithms/search${qs ? `?${qs}` : ""}`);
  }

  /**
   * Approve or reject an algorithm.
   *
   * @param algorithmId - The algorithm to review.
   * @param status - The review decision: "approved" or "rejected".
   */
  async review(
    algorithmId: string,
    status: "approved" | "rejected",
  ): Promise<AlgorithmReviewResponse> {
    return this._post(`/api/v1/algorithms/${algorithmId}/review`, { status });
  }

  /**
   * Submit reviewed algorithms — queue embedding for approved algorithms.
   *
   * @param documentId - The document whose algorithms to submit.
   */
  async submit(documentId: string): Promise<AlgorithmSubmitResponse> {
    return this._post(
      `/api/v1/algorithms/by-document/${documentId}/submit`,
      {},
    );
  }

  /**
   * Delete a single algorithm by ID.
   *
   * @param algorithmId - The algorithm to delete.
   */
  async delete(algorithmId: string): Promise<AlgorithmDeleteResponse> {
    return this._del(`/api/v1/algorithms/${algorithmId}`);
  }

  /**
   * Delete all algorithms extracted from a document.
   *
   * @param documentId - The document whose algorithms should be deleted.
   */
  async deleteByDocument(
    documentId: string,
  ): Promise<AlgorithmDeleteResponse> {
    return this._del(`/api/v1/algorithms/by-document/${documentId}`);
  }
}
