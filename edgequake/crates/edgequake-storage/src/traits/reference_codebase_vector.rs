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
        algorithm_ids: Option<&[Uuid]>,
    ) -> Result<Vec<ReferenceCodebaseSearchHit>>;
}
