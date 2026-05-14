//! Vector storage trait for similarity search.
//!
//! # Implements
//!
//! - **FEAT0201**: Vector Similarity Search
//!
//! # Enforces
//!
//! - **BR0201**: Namespace-based tenant isolation
//! - **BR0010**: Embedding dimension validated on insert
//!
//! # WHY: Separate Vector Storage
//!
//! Vector similarity search is specialized:
//! - Requires optimized index structures (HNSW, IVF)
//! - Benefits from GPU acceleration
//! - Different scaling characteristics than graph/KV
//!
//! Abstracting as a trait allows using:
//! - pgvector (PostgreSQL extension)
//! - Pinecone, Weaviate, Qdrant (managed services)
//! - In-memory brute-force (testing)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Vector similarity search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// Record identifier
    pub id: String,
    /// Similarity score (higher is more similar)
    pub score: f32,
    /// Associated metadata
    pub metadata: serde_json::Value,
}

/// Metadata-based filter for vector queries (SPEC-007 Tier 2+).
///
/// All fields are optional; only non-None fields participate in AND-combined filtering.
/// Pushes filtering to the SQL layer (JSONB WHERE or column WHERE) to avoid
/// retrieving and discarding irrelevant vectors in application code.
///
/// @implements SPEC-007 R-T2-01
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataFilter {
    /// Filter by document ID(s). Matches JSONB key `document_id` OR `source_document_id`.
    pub document_ids: Option<Vec<String>>,
    /// Filter by tenant ID.
    pub tenant_id: Option<String>,
    /// Filter by workspace ID.
    pub workspace_id: Option<String>,
    /// Filter by `metadata->>'type'` (chunk / entity / relationship / algorithm).
    /// Pushed into the SQL WHERE so HNSW's top-K candidate pool is drawn
    /// only from rows of that type — critical when the target type is rare
    /// compared to chunks/entities and would otherwise be evicted from
    /// HNSW's bounded candidate set before reaching the filter.
    pub vector_type: Option<String>,
}

impl MetadataFilter {
    /// Returns true when no filter fields are set.
    pub fn is_empty(&self) -> bool {
        self.document_ids.is_none()
            && self.tenant_id.is_none()
            && self.workspace_id.is_none()
            && self.vector_type.is_none()
    }

    /// Build a filter from optional tenant and workspace IDs.
    pub fn from_tenant_workspace(
        tenant_id: Option<String>,
        workspace_id: Option<String>,
    ) -> Option<Self> {
        if tenant_id.is_none() && workspace_id.is_none() {
            return None;
        }
        Some(Self {
            document_ids: None,
            tenant_id,
            workspace_id,
            vector_type: None,
        })
    }

    /// Build a filter from tenant + workspace + an explicit `vector_type` to
    /// push into the SQL WHERE clause. Used by the naive-chunk retrieval arm
    /// in the workspace-scoped hybrid path so HNSW's top-K candidate pool is
    /// drawn only from chunk rows — otherwise large graphs (60k+ entities)
    /// dominate the candidate pool and zero chunks survive in-memory
    /// filtering (the original OODA-230 failure mode).
    pub fn from_tenant_workspace_type(
        tenant_id: Option<String>,
        workspace_id: Option<String>,
        vector_type: impl Into<String>,
    ) -> Option<Self> {
        Some(Self {
            document_ids: None,
            tenant_id,
            workspace_id,
            vector_type: Some(vector_type.into()),
        })
    }
}

/// Vector storage interface for similarity search.
///
/// Provides storage and retrieval of vector embeddings with
/// support for similarity search operations.
///
/// # Implementations
///
/// - `MemoryVectorStorage` - In-memory brute-force search (testing)
/// - `PgVectorStorage` - PostgreSQL with pgvector extension
/// - `SurrealDBVectorStorage` - SurrealDB native vector support
#[async_trait]
pub trait VectorStorage: Send + Sync {
    /// Get the storage namespace.
    fn namespace(&self) -> &str;

    /// Get the expected embedding dimension.
    fn dimension(&self) -> usize;

    /// Initialize the vector storage.
    ///
    /// Creates necessary indices and tables.
    async fn initialize(&self) -> Result<()>;

    /// Flush pending changes.
    async fn finalize(&self) -> Result<()>;

    /// Perform similarity search.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The query vector
    /// * `top_k` - Maximum number of results to return
    /// * `filter_ids` - Optional list of IDs to restrict search to
    ///
    /// # Returns
    ///
    /// Vector of search results ordered by similarity (highest first).
    async fn query(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter_ids: Option<&[String]>,
    ) -> Result<Vec<VectorSearchResult>>;

    /// Insert or update vectors with metadata.
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of (id, embedding, metadata) tuples
    async fn upsert(&self, data: &[(String, Vec<f32>, serde_json::Value)]) -> Result<()>;

    /// Delete vectors by IDs.
    async fn delete(&self, ids: &[String]) -> Result<()>;

    /// Delete all vectors associated with an entity.
    ///
    /// This is used when deleting an entity to clean up its embeddings.
    async fn delete_entity(&self, entity_name: &str) -> Result<()>;

    /// Delete all relationship vectors involving an entity.
    ///
    /// Used when cascading entity deletion.
    async fn delete_entity_relations(&self, entity_name: &str) -> Result<()>;

    /// Delete every vector row (entity + chunk + anything else) that
    /// belongs to the given document.
    ///
    /// Used on the reprocess / re-ingest path: without this, a second
    /// extraction accumulates on top of the first — old entities keep
    /// surfacing in RAG retrieval and graph views even though the new
    /// extraction no longer emits them. Deletion is keyed on the
    /// `document_id` metadata column so it catches rows the graph-
    /// traversal cleanup can't (e.g. entities the current extraction
    /// wouldn't re-emit, so their IDs don't appear in the fresh
    /// graph).
    ///
    /// Default implementation returns 0 — adapters that don't track
    /// document_id opt out and the reprocess path silently skips them.
    /// PostgreSQL workspace vectors override this.
    ///
    /// Returns the number of rows deleted.
    async fn delete_by_document_id(&self, document_id: &str) -> Result<usize> {
        let _ = document_id;
        Ok(0)
    }

    /// Get a single vector by ID.
    async fn get_by_id(&self, id: &str) -> Result<Option<Vec<f32>>>;

    /// Get multiple vectors by IDs.
    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<(String, Vec<f32>)>>;

    /// Check if storage is empty.
    async fn is_empty(&self) -> Result<bool>;

    /// Get count of stored vectors.
    async fn count(&self) -> Result<usize>;

    /// Clear all vectors.
    async fn clear(&self) -> Result<()>;

    /// Clear vectors for a specific workspace.
    ///
    /// This is used when rebuilding embeddings for a single workspace
    /// without affecting other workspaces.
    ///
    /// # Arguments
    ///
    /// * `workspace_id` - The UUID of the workspace to clear vectors for
    ///
    /// # Returns
    ///
    /// Number of vectors deleted.
    ///
    /// # Default Implementation
    ///
    /// Returns 0 by default. Implementations should override this for
    /// workspace-scoped clearing.
    async fn clear_workspace(&self, workspace_id: &uuid::Uuid) -> Result<usize> {
        // Default implementation does nothing - clear() clears all
        // Implementations should override this for workspace-scoped clearing
        let _ = workspace_id;
        Ok(0)
    }

    /// Query with metadata pre-filter (SPEC-007 Tier 2+).
    ///
    /// Pushes tenant/workspace/document filters to the storage layer (SQL WHERE)
    /// instead of post-filtering in application code.
    ///
    /// Default implementation ignores `metadata_filter` and delegates to `query()`.
    /// Backends that support SQL-level filtering override this for better performance.
    ///
    /// @implements SPEC-007 R-T2-01
    async fn query_filtered(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter_ids: Option<&[String]>,
        metadata_filter: Option<&MetadataFilter>,
    ) -> Result<Vec<VectorSearchResult>> {
        let _ = metadata_filter;
        self.query(query_embedding, top_k, filter_ids).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_filter_is_empty() {
        let f = MetadataFilter::default();
        assert!(f.is_empty());

        let f = MetadataFilter {
            tenant_id: Some("t1".into()),
            ..Default::default()
        };
        assert!(!f.is_empty());
    }

    #[test]
    fn test_metadata_filter_from_tenant_workspace_both_none() {
        assert!(MetadataFilter::from_tenant_workspace(None, None).is_none());
    }

    #[test]
    fn test_metadata_filter_from_tenant_workspace_tenant_only() {
        let mf = MetadataFilter::from_tenant_workspace(Some("t1".into()), None).unwrap();
        assert_eq!(mf.tenant_id.as_deref(), Some("t1"));
        assert!(mf.workspace_id.is_none());
        assert!(mf.document_ids.is_none());
    }

    #[test]
    fn test_metadata_filter_from_tenant_workspace_both() {
        let mf =
            MetadataFilter::from_tenant_workspace(Some("t1".into()), Some("ws1".into())).unwrap();
        assert_eq!(mf.tenant_id.as_deref(), Some("t1"));
        assert_eq!(mf.workspace_id.as_deref(), Some("ws1"));
    }

    #[test]
    fn test_metadata_filter_serialization_roundtrip() {
        let mf = MetadataFilter {
            document_ids: Some(vec!["doc1".into(), "doc2".into()]),
            tenant_id: Some("t1".into()),
            workspace_id: Some("ws1".into()),
            vector_type: None,
        };
        let json = serde_json::to_string(&mf).unwrap();
        let mf2: MetadataFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(mf2.tenant_id, mf.tenant_id);
        assert_eq!(mf2.workspace_id, mf.workspace_id);
        assert_eq!(mf2.document_ids, mf.document_ids);
    }
}
