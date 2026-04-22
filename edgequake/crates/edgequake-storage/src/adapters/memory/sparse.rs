//! In-memory sparse chunk storage for tests and lightweight deployments.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::traits::{
    MetadataFilter, SparseChunkDocument, SparseChunkSearchResult, SparseChunkStorage,
};

/// In-memory BM25-style sparse chunk storage.
pub struct MemorySparseChunkStorage {
    chunks: RwLock<HashMap<String, SparseChunkDocument>>,
}

impl MemorySparseChunkStorage {
    /// Create an empty memory sparse index.
    pub fn new() -> Self {
        Self {
            chunks: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemorySparseChunkStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SparseChunkStorage for MemorySparseChunkStorage {
    async fn initialize(&self) -> Result<()> {
        Ok(())
    }

    async fn upsert_chunks(&self, chunks: &[SparseChunkDocument]) -> Result<()> {
        let mut guard = self.chunks.write().await;
        for chunk in chunks {
            guard.insert(chunk.id.clone(), chunk.clone());
        }
        Ok(())
    }

    async fn search_chunks_bm25(
        &self,
        query: &str,
        top_k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SparseChunkSearchResult>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let guard = self.chunks.read().await;
        let corpus: Vec<&SparseChunkDocument> = guard
            .values()
            .filter(|c| matches_filter(c, filter))
            .collect();
        let query_terms = tokenize(query);
        if corpus.is_empty() || query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let avg_len = corpus
            .iter()
            .map(|doc| tokenize(&doc.content).len())
            .sum::<usize>() as f32
            / corpus.len() as f32;

        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut doc_terms: HashMap<&str, Vec<String>> = HashMap::new();
        for doc in &corpus {
            let terms = tokenize(&doc.content);
            let unique: HashSet<_> = terms.iter().cloned().collect();
            for term in unique {
                *doc_freq.entry(term).or_insert(0) += 1;
            }
            doc_terms.insert(&doc.id, terms);
        }

        let mut results = Vec::new();
        for doc in corpus {
            let Some(terms) = doc_terms.get(doc.id.as_str()) else {
                continue;
            };
            let score = bm25_score(&query_terms, terms, &doc_freq, guard.len(), avg_len);
            if score > 0.0 {
                results.push(SparseChunkSearchResult {
                    id: doc.id.clone(),
                    score,
                    metadata: doc.metadata.clone(),
                });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        Ok(results)
    }

    async fn delete_by_document_id(&self, document_id: &str) -> Result<usize> {
        let mut guard = self.chunks.write().await;
        let before = guard.len();
        guard.retain(|_, chunk| metadata_str(&chunk.metadata, "document_id") != Some(document_id));
        Ok(before - guard.len())
    }

    async fn delete_chunks(&self, ids: &[String]) -> Result<()> {
        let mut guard = self.chunks.write().await;
        for id in ids {
            guard.remove(id);
        }
        Ok(())
    }

    async fn clear_workspace(&self, workspace_id: &str) -> Result<usize> {
        let mut guard = self.chunks.write().await;
        let before = guard.len();
        guard
            .retain(|_, chunk| metadata_str(&chunk.metadata, "workspace_id") != Some(workspace_id));
        Ok(before - guard.len())
    }

    async fn clear(&self) -> Result<()> {
        self.chunks.write().await.clear();
        Ok(())
    }
}

fn matches_filter(chunk: &SparseChunkDocument, filter: Option<&MetadataFilter>) -> bool {
    let Some(filter) = filter else {
        return true;
    };

    if let Some(ids) = &filter.document_ids {
        let document_id = metadata_str(&chunk.metadata, "document_id")
            .or_else(|| metadata_str(&chunk.metadata, "source_document_id"));
        if !document_id.is_some_and(|id| ids.iter().any(|allowed| allowed == id)) {
            return false;
        }
    }
    if let Some(tenant_id) = &filter.tenant_id {
        if metadata_str(&chunk.metadata, "tenant_id") != Some(tenant_id.as_str()) {
            return false;
        }
    }
    if let Some(workspace_id) = &filter.workspace_id {
        if metadata_str(&chunk.metadata, "workspace_id") != Some(workspace_id.as_str()) {
            return false;
        }
    }
    if let Some(vector_type) = &filter.vector_type {
        if metadata_str(&chunk.metadata, "type") != Some(vector_type.as_str()) {
            return false;
        }
    }
    true
}

fn metadata_str<'a>(metadata: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    metadata.get(key).and_then(|v| v.as_str())
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|token| {
            token
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|token| token.len() > 1)
        .collect()
}

fn bm25_score(
    query_terms: &[String],
    doc_terms: &[String],
    doc_freq: &HashMap<String, usize>,
    total_docs: usize,
    avg_len: f32,
) -> f32 {
    let k1 = 1.2;
    let b = 0.75;
    let doc_len = doc_terms.len().max(1) as f32;
    let avg_len = avg_len.max(1.0);
    let mut score = 0.0;
    let mut seen = HashSet::new();

    for term in query_terms {
        if !seen.insert(term) {
            continue;
        }
        let tf = doc_terms
            .iter()
            .filter(|candidate| *candidate == term)
            .count() as f32;
        if tf == 0.0 {
            continue;
        }
        let df = *doc_freq.get(term).unwrap_or(&0) as f32;
        let idf = ((total_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
        let numerator = tf * (k1 + 1.0);
        let denominator = tf + k1 * (1.0 - b + b * doc_len / avg_len);
        score += idf * numerator / denominator;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn filters_by_workspace() {
        let storage = MemorySparseChunkStorage::new();
        storage
            .upsert_chunks(&[
                SparseChunkDocument::new(
                    "a",
                    "Euston station",
                    json!({"type": "chunk", "workspace_id": "ws1"}),
                ),
                SparseChunkDocument::new(
                    "b",
                    "Euston station",
                    json!({"type": "chunk", "workspace_id": "ws2"}),
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
