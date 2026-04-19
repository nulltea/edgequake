/**
 * Reference-codebase resource — Phase 2 of the Reference Code GraphRAG
 * extension. A coding agent working on porting or integrating a paper's
 * reference implementation can:
 *
 *  - `query(...)`   — semantic search over approved repo chunks.
 *  - `graph(...)`   — BFS the symbol graph around a known anchor.
 *
 * @module resources/reference-codebase
 * @see edgequake/crates/edgequake-api/src/handlers/reference_codebase/mod.rs
 */

import type {
  CreateReferenceCodebaseIndexRequest,
  CreateReferenceCodebaseIndexResponse,
  ReferenceCodebaseGraphParams,
  ReferenceCodebaseGraphResponse,
  ReferenceCodebaseIndex,
  ReferenceCodebaseQueryRequest,
  ReferenceCodebaseQueryResponse,
} from "../types/reference-codebase.js";
import { Resource } from "./base.js";

export class ReferenceCodebaseResource extends Resource {
  /** Enqueue indexing for an approved repo. 202 with track_id. */
  async createIndex(
    request: CreateReferenceCodebaseIndexRequest,
  ): Promise<CreateReferenceCodebaseIndexResponse> {
    return this._post("/api/v1/reference-codebase/indexes", request);
  }

  /** Status + counts for a specific index. */
  async getIndex(indexId: string): Promise<ReferenceCodebaseIndex> {
    return this._get(`/api/v1/reference-codebase/indexes/${indexId}`);
  }

  /** Semantic search over approved code chunks. */
  async query(
    request: ReferenceCodebaseQueryRequest,
  ): Promise<ReferenceCodebaseQueryResponse> {
    return this._post("/api/v1/reference-codebase/query", request);
  }

  /**
   * Anchor-centric BFS subgraph. Pass exactly one of
   * `anchor_artifact_id` or `anchor_symbol`; omit both to get the
   * top-degree fallback (useful for exploring `mode='full'` indexes
   * without a known entry point).
   */
  async graph(
    indexId: string,
    params: ReferenceCodebaseGraphParams = {},
  ): Promise<ReferenceCodebaseGraphResponse> {
    const q = new URLSearchParams();
    if (params.anchor_artifact_id) q.set("anchor_artifact_id", params.anchor_artifact_id);
    if (params.anchor_symbol) q.set("anchor_symbol", params.anchor_symbol);
    if (params.hops !== undefined) q.set("hops", String(params.hops));
    if (params.max_nodes !== undefined) q.set("max_nodes", String(params.max_nodes));
    const qs = q.toString();
    const path = `/api/v1/reference-codebase/indexes/${indexId}/graph${qs ? `?${qs}` : ""}`;
    return this._get(path);
  }
}
