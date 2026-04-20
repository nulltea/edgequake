//! Vector search over approved `algorithms` surfaced via semantic similarity.
//!
//! Sibling to [`CodeVectorStorage`]. Algorithm embeddings live in the
//! **workspace** vector store tagged with `metadata.type = "algorithm"` and
//! `metadata.algorithm_id = <uuid>`. The main query filter only lets
//! `chunk/entity/relationship` rows through, so these embeddings are
//! effectively dead unless reached via this dedicated path.
//!
//! The impl is handed the workspace's `VectorStorage` at call time because
//! — unlike code, which has its own dedicated pgvector table — algorithm
//! embeddings share the workspace's main vector table and its dimension
//! varies per workspace. The trait keeps the caller unaware of the SQL and
//! of the `algorithms` table lookup.
//!
//! [`CodeVectorStorage`]: crate::traits::CodeVectorStorage

use async_trait::async_trait;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::Result;
use crate::traits::VectorStorage;

/// Per-step summary as rendered in the LLM prompt.
#[derive(Debug, Clone)]
pub struct AlgorithmStepSummary {
    pub number: usize,
    pub action: String,
    pub details: String,
}

/// One approved algorithm returned by
/// [`AlgorithmVectorStorage::search_approved_algorithms`]. Structured so the
/// renderer can emit pseudocode, step-by-step, or a compact summary without
/// re-querying Postgres.
#[derive(Debug, Clone)]
pub struct AlgorithmSearchHit {
    pub algorithm_id: String,
    pub document_id: String,
    pub name: String,
    pub description: Option<String>,
    pub algorithm_type: String,
    pub pseudocode: Option<String>,
    pub complexity: Option<String>,
    pub steps: Vec<AlgorithmStepSummary>,
    pub tags: Vec<String>,
    pub confidence: String,
    /// Cosine distance in `[0, 2]`. Lower is more similar. Derived from the
    /// workspace vector store's similarity score (`distance = 1 - score`).
    pub cosine_distance: f64,
}

/// Vector search over approved `algorithms`, routed through the workspace's
/// main vector store plus a Postgres join for structured fields.
#[async_trait]
pub trait AlgorithmVectorStorage: Send + Sync {
    /// Return the nearest approved algorithms to `query_embedding` within
    /// `max_distance`, scoped to `(tenant_id, workspace_id)` and optionally
    /// restricted to `document_ids`.
    ///
    /// `workspace_vectors` is the per-workspace `VectorStorage` handle the
    /// query already holds — the impl uses it so the search honours the
    /// workspace's embedding dimension and table.
    async fn search_approved_algorithms(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        workspace_vectors: &Arc<dyn VectorStorage>,
        query_embedding: &[f32],
        limit: i64,
        max_distance: f64,
        document_ids: Option<&[String]>,
    ) -> Result<Vec<AlgorithmSearchHit>>;
}
