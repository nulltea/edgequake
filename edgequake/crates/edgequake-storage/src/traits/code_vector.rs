//! Vector search over approved code_artifacts.
//!
//! Phase 1 of the Reference Code GraphRAG extension ships code embeddings in
//! a *separate* pgvector table from the document-text workspace stores,
//! because the code embedder (Jina code-embeddings, 896-d) has a different
//! dimension from the text embedder per workspace. That table is an
//! implementation detail — callers see only the trait.

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;

/// One hit returned by [`CodeVectorStorage::search_approved_code`].
///
/// Pure data — no `edgequake-query` types leak in, so the trait stays
/// reusable by anything that needs approved code retrieval.
#[derive(Debug, Clone)]
pub struct CodeSearchHit {
    pub algorithm_id: String,
    pub algorithm_name: String,
    pub document_id: String,
    pub file_path: String,
    pub start_line: i32,
    pub end_line: i32,
    pub language: String,
    pub snippet: String,
    pub repo_url: Option<String>,
    pub repo_commit: String,
    pub match_rationale: Option<String>,
    /// Cosine distance in `[0, 2]`. Lower is more similar.
    pub cosine_distance: f64,
}

/// Vector search over approved `code_artifacts`.
#[async_trait]
pub trait CodeVectorStorage: Send + Sync {
    /// Return the nearest approved snippets to `query_vec` within
    /// `max_distance`, scoped to `(tenant_id, workspace_id)`.
    ///
    /// When `document_ids` is `Some(&[...])`, hits are restricted to code
    /// artifacts belonging to those documents — used by the query engine
    /// to avoid cross-citing code from papers that weren't actually
    /// surfaced by the main retrieval pass. `None` (or an empty slice)
    /// disables the filter and searches the whole workspace.
    async fn search_approved_code(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        query_vec: &[f32],
        limit: i64,
        max_distance: f64,
        document_ids: Option<&[String]>,
    ) -> Result<Vec<CodeSearchHit>>;
}
