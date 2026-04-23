//! Vector search over Phase 2 reference-codebase chunks.

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;

#[derive(Debug, Clone)]
pub struct ReferenceCodebaseSearchHit {
    pub chunk_id: Uuid,
    pub index_id: Uuid,
    pub document_id: String,
    pub document_repo_id: Uuid,
    pub repo_url: String,
    pub repo_commit: String,
    pub file_path: String,
    pub language: String,
    pub symbol_name: Option<String>,
    pub start_line: i32,
    pub end_line: i32,
    pub chunk_kind: String,
    pub algorithm_id: Option<Uuid>,
    pub content: String,
    pub cosine_distance: f64,
    /// When populated, the hit was surfaced by entity expansion (exact symbol-name
    /// match from the query text) rather than vector similarity. Lets clients
    /// display a "matched X" affordance and understand why the hit ranked first.
    pub matched_entity: Option<String>,
    /// BM25 score (normalized [0, 1]) for this hit against the query, computed
    /// in the handler layer after the storage call. `None` when BM25 isn't
    /// applicable (e.g. empty query).
    pub bm25_score: Option<f64>,
    /// Final blended score used for ordering: `0.6*(1 - cosine) + 0.4*bm25 +
    /// entity_boost`. Clients can rank by this directly; `cosine_distance`
    /// stays the raw vector value for compatibility.
    pub final_score: Option<f64>,
}

#[async_trait]
pub trait ReferenceCodebaseVectorStorage: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    async fn search_reference_codebase(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        query_vec: &[f32],
        limit: i64,
        max_distance: f64,
        document_repo_id: Option<Uuid>,
        index_id: Option<Uuid>,
        algorithm_context_ids: Option<&[Uuid]>,
    ) -> Result<Vec<ReferenceCodebaseSearchHit>>;

    /// Fetch chunks whose `symbol_name` matches one of the given candidates.
    /// Hits carry `matched_entity` set to the matched candidate and
    /// `cosine_distance = 0.0` — callers merge these ahead of vector hits so
    /// the agent sees exact-name matches first.
    async fn fetch_chunks_by_symbol_names(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        names: &[String],
        document_repo_id: Option<Uuid>,
        index_id: Option<Uuid>,
        algorithm_context_ids: Option<&[Uuid]>,
        limit: i64,
    ) -> Result<Vec<ReferenceCodebaseSearchHit>>;
}
