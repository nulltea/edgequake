//! Tantivy-backed sparse BM25 storage for chunk retrieval.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::tokenizer::{Language, LowerCaser, SimpleTokenizer, Stemmer, TextAnalyzer};
use tantivy::{doc, Index, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use crate::error::{Result, StorageError};
use crate::traits::{
    MetadataFilter, SparseChunkDocument, SparseChunkSearchResult, SparseChunkStorage,
};

const CONTENT_TOKENIZER: &str = "edgequake_english_stem";
const WRITER_HEAP_BYTES: usize = 50_000_000;

/// Persistent BM25 sparse index over document chunks.
pub struct TantivySparseChunkStorage {
    index: Index,
    id_field: Field,
    content_field: Field,
    metadata_field: Field,
    document_id_field: Field,
    tenant_id_field: Field,
    workspace_id_field: Field,
    type_field: Field,
    index_path: PathBuf,
    writer_lock: Mutex<()>,
}

impl TantivySparseChunkStorage {
    /// Create or open a sparse chunk index at `index_path`.
    pub fn new(index_path: impl AsRef<Path>) -> Result<Self> {
        let index_path = index_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&index_path)?;

        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_text_field("id", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", Self::content_text_options());
        let metadata_field = schema_builder.add_text_field("metadata", STORED);
        let document_id_field = schema_builder.add_text_field("document_id", STRING | STORED);
        let tenant_id_field = schema_builder.add_text_field("tenant_id", STRING | STORED);
        let workspace_id_field = schema_builder.add_text_field("workspace_id", STRING | STORED);
        let type_field = schema_builder.add_text_field("type", STRING | STORED);
        let schema = schema_builder.build();

        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(&index_path)
                .map_err(|e| StorageError::Database(format!("Failed to open BM25 index: {e}")))?
        } else {
            Index::create_in_dir(&index_path, schema)
                .map_err(|e| StorageError::Database(format!("Failed to create BM25 index: {e}")))?
        };
        Self::register_tokenizers(&index);

        Ok(Self {
            index,
            id_field,
            content_field,
            metadata_field,
            document_id_field,
            tenant_id_field,
            workspace_id_field,
            type_field,
            index_path,
            writer_lock: Mutex::new(()),
        })
    }

    fn register_tokenizers(index: &Index) {
        let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .filter(Stemmer::new(Language::English))
            .build();
        index.tokenizers().register(CONTENT_TOKENIZER, analyzer);
    }

    fn content_text_options() -> TextOptions {
        let indexing = TextFieldIndexing::default()
            .set_tokenizer(CONTENT_TOKENIZER)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        TextOptions::default().set_indexing_options(indexing)
    }

    fn writer(&self) -> Result<IndexWriter<TantivyDocument>> {
        self.index
            .writer(WRITER_HEAP_BYTES)
            .map_err(|e| StorageError::Database(format!("Failed to create BM25 writer: {e}")))
    }

    fn stored_text(doc: &TantivyDocument, field: Field) -> Option<String> {
        doc.get_first(field)
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
    }

    fn matches_filter_doc(&self, doc: &TantivyDocument, filter: Option<&MetadataFilter>) -> bool {
        let Some(filter) = filter else {
            return true;
        };

        if let Some(ids) = &filter.document_ids {
            let doc_id = Self::stored_text(doc, self.document_id_field);
            if !doc_id
                .as_ref()
                .is_some_and(|candidate| ids.iter().any(|allowed| allowed == candidate))
            {
                return false;
            }
        }
        if let Some(tenant_id) = &filter.tenant_id {
            if Self::stored_text(doc, self.tenant_id_field).as_deref() != Some(tenant_id) {
                return false;
            }
        }
        if let Some(workspace_id) = &filter.workspace_id {
            if Self::stored_text(doc, self.workspace_id_field).as_deref() != Some(workspace_id) {
                return false;
            }
        }
        if let Some(vector_type) = &filter.vector_type {
            if Self::stored_text(doc, self.type_field).as_deref() != Some(vector_type) {
                return false;
            }
        }
        true
    }

    fn document_id(metadata: &serde_json::Value) -> String {
        metadata
            .get("document_id")
            .or_else(|| metadata.get("source_document_id"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    }

    fn metadata_str(metadata: &serde_json::Value, key: &str) -> String {
        metadata
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    }

    fn is_lock_stale(lock_path: &Path) -> bool {
        lock_path
            .metadata()
            .and_then(|metadata| metadata.modified())
            .and_then(|modified| modified.elapsed().map_err(std::io::Error::other))
            .map(|elapsed| elapsed.as_secs() > 300)
            .unwrap_or(false)
    }

    fn cleanup_stale_locks(&self) {
        for filename in [".tantivy-writer.lock", ".tantivy-meta.lock"] {
            let path = self.index_path.join(filename);
            if Self::is_lock_stale(&path) {
                if let Err(error) = std::fs::remove_file(&path) {
                    tracing::warn!(?path, %error, "Failed to remove stale BM25 lock file");
                }
            }
        }
    }
}

#[async_trait]
impl SparseChunkStorage for TantivySparseChunkStorage {
    async fn initialize(&self) -> Result<()> {
        Ok(())
    }

    async fn upsert_chunks(&self, chunks: &[SparseChunkDocument]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| StorageError::Database(format!("BM25 writer lock poisoned: {e}")))?;
        self.cleanup_stale_locks();
        let mut writer = self.writer()?;

        for chunk in chunks {
            writer.delete_term(Term::from_field_text(self.id_field, &chunk.id));
            let metadata_json = serde_json::to_string(&chunk.metadata)?;
            let document_id = Self::document_id(&chunk.metadata);
            let tenant_id = Self::metadata_str(&chunk.metadata, "tenant_id");
            let workspace_id = Self::metadata_str(&chunk.metadata, "workspace_id");
            let vector_type = Self::metadata_str(&chunk.metadata, "type");

            writer
                .add_document(doc!(
                    self.id_field => chunk.id.clone(),
                    self.content_field => chunk.content.clone(),
                    self.metadata_field => metadata_json,
                    self.document_id_field => document_id,
                    self.tenant_id_field => tenant_id,
                    self.workspace_id_field => workspace_id,
                    self.type_field => vector_type,
                ))
                .map_err(|e| StorageError::Database(format!("Failed to add BM25 doc: {e}")))?;
        }

        writer
            .commit()
            .map_err(|e| StorageError::Database(format!("Failed to commit BM25 docs: {e}")))?;
        Ok(())
    }

    async fn search_chunks_bm25(
        &self,
        query: &str,
        top_k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SparseChunkSearchResult>> {
        if top_k == 0 || query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| StorageError::Database(format!("Failed to create BM25 reader: {e}")))?;
        reader
            .reload()
            .map_err(|e| StorageError::Database(format!("Failed to reload BM25 reader: {e}")))?;

        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        let (parsed, errors) = query_parser.parse_query_lenient(query);
        if !errors.is_empty() {
            tracing::debug!(
                ?errors,
                query,
                "BM25 query parser recovered from syntax errors"
            );
        }

        let candidate_limit = (top_k * 20).max(top_k).min(1000);
        let top_docs = searcher
            .search(&parsed, &TopDocs::with_limit(candidate_limit))
            .map_err(|e| StorageError::Database(format!("Failed to search BM25 index: {e}")))?;

        let mut results = Vec::with_capacity(top_k);
        for (score, address) in top_docs {
            let doc: TantivyDocument = searcher.doc(address).map_err(|e| {
                StorageError::Database(format!("Failed to load BM25 document: {e}"))
            })?;

            if !self.matches_filter_doc(&doc, filter) {
                continue;
            }

            let Some(id) = Self::stored_text(&doc, self.id_field) else {
                continue;
            };
            let metadata = Self::stored_text(&doc, self.metadata_field)
                .and_then(|raw| serde_json::from_str(&raw).ok())
                .unwrap_or_else(|| serde_json::json!({}));

            results.push(SparseChunkSearchResult {
                id,
                score,
                metadata,
            });
            if results.len() >= top_k {
                break;
            }
        }

        Ok(results)
    }

    async fn delete_by_document_id(&self, document_id: &str) -> Result<usize> {
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| StorageError::Database(format!("BM25 writer lock poisoned: {e}")))?;
        let mut writer = self.writer()?;
        writer.delete_term(Term::from_field_text(self.document_id_field, document_id));
        writer
            .commit()
            .map_err(|e| StorageError::Database(format!("Failed to commit BM25 deletion: {e}")))?;
        Ok(0)
    }

    async fn delete_chunks(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| StorageError::Database(format!("BM25 writer lock poisoned: {e}")))?;
        let mut writer = self.writer()?;
        for id in ids {
            writer.delete_term(Term::from_field_text(self.id_field, id));
        }
        writer
            .commit()
            .map_err(|e| StorageError::Database(format!("Failed to commit BM25 deletion: {e}")))?;
        Ok(())
    }

    async fn clear_workspace(&self, workspace_id: &str) -> Result<usize> {
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| StorageError::Database(format!("BM25 writer lock poisoned: {e}")))?;
        let mut writer = self.writer()?;
        writer.delete_term(Term::from_field_text(self.workspace_id_field, workspace_id));
        writer
            .commit()
            .map_err(|e| StorageError::Database(format!("Failed to commit BM25 deletion: {e}")))?;
        Ok(0)
    }

    async fn clear(&self) -> Result<()> {
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| StorageError::Database(format!("BM25 writer lock poisoned: {e}")))?;
        let mut writer = self.writer()?;
        writer
            .delete_all_documents()
            .map_err(|e| StorageError::Database(format!("Failed to clear BM25 index: {e}")))?;
        writer
            .commit()
            .map_err(|e| StorageError::Database(format!("Failed to commit BM25 clear: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[tokio::test]
    async fn searches_exact_terms_and_filters_workspace() {
        let dir = tempdir().unwrap();
        let storage = TantivySparseChunkStorage::new(dir.path()).unwrap();
        storage
            .upsert_chunks(&[
                SparseChunkDocument::new(
                    "a",
                    "Euston station privacy",
                    json!({"type": "chunk", "workspace_id": "ws1", "content": "Euston station privacy"}),
                ),
                SparseChunkDocument::new(
                    "b",
                    "Euston unrelated workspace",
                    json!({"type": "chunk", "workspace_id": "ws2", "content": "Euston unrelated workspace"}),
                ),
            ])
            .await
            .unwrap();

        let filter = MetadataFilter {
            workspace_id: Some("ws1".to_string()),
            ..Default::default()
        };
        let results = storage
            .search_chunks_bm25("Euston", 10, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }
}
