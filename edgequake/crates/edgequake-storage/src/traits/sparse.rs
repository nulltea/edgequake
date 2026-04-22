//! Sparse lexical retrieval for document chunks.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::traits::MetadataFilter;

/// Chunk document stored in the sparse lexical index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseChunkDocument {
    /// Chunk identifier.
    pub id: String,
    /// Chunk text used for BM25 indexing.
    pub content: String,
    /// Full chunk metadata used to hydrate query context.
    pub metadata: serde_json::Value,
}

impl SparseChunkDocument {
    /// Build a sparse chunk document.
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata,
        }
    }
}

/// Result returned by sparse lexical chunk search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseChunkSearchResult {
    /// Chunk identifier.
    pub id: String,
    /// BM25 score returned by the sparse backend.
    pub score: f32,
    /// Stored chunk metadata.
    pub metadata: serde_json::Value,
}

/// Sparse retrieval interface for corpus-level lexical chunk search.
#[async_trait]
pub trait SparseChunkStorage: Send + Sync {
    /// Initialize underlying sparse index resources.
    async fn initialize(&self) -> Result<()>;

    /// Insert or replace chunk documents.
    async fn upsert_chunks(&self, chunks: &[SparseChunkDocument]) -> Result<()>;

    /// Search chunk content with BM25-style lexical scoring.
    async fn search_chunks_bm25(
        &self,
        query: &str,
        top_k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SparseChunkSearchResult>>;

    /// Delete indexed chunks by document ID.
    async fn delete_by_document_id(&self, document_id: &str) -> Result<usize>;

    /// Delete indexed chunks by IDs.
    async fn delete_chunks(&self, ids: &[String]) -> Result<()>;

    /// Clear all indexed chunks for a workspace.
    async fn clear_workspace(&self, workspace_id: &str) -> Result<usize>;

    /// Clear all indexed chunks.
    async fn clear(&self) -> Result<()>;
}
