//! Reranking step.
//!
//! Rescores in-memory candidate chunks against the query, selecting between
//! BM25 (in-process) and a cross-encoder HTTP reranker based on the
//! workspace-level `reranker_strategy`. `"off"` short-circuits the step;
//! `None` falls back to the engine default (BM25). The HTTP path is wired
//! by `state/mod.rs::create_semantic_config`.

use std::sync::Arc;

use super::SOTAQueryEngine;
use crate::error::Result;

impl SOTAQueryEngine {
    /// Re-order `chunks` by reranker relevance to `query`.
    ///
    /// * `strategy` — `Some("off")` skips reranking entirely;
    ///   `Some("semantic")` picks the cross-encoder (with `model` as the
    ///   model-name override); anything else (including `None`) uses BM25.
    /// * Output is truncated to `config.rerank_top_k`.
    /// * Chunks below `config.min_rerank_score` are dropped, with a
    ///   recall-preserving fallback to the original top-K if every
    ///   candidate falls under the floor.
    /// * Reranker errors propagate (hard-fail) — semantic API blips don't
    ///   silently degrade to unranked.
    /// * `SOTAQueryConfig::enable_rerank` is an engine-internal kill switch
    ///   for test/dev environments; production callers should use the
    ///   per-workspace `reranker_strategy = "off"` instead.
    pub(super) async fn rerank_chunks_with_strategy(
        &self,
        query: &str,
        mut chunks: Vec<crate::context::RetrievedChunk>,
        strategy: Option<&str>,
        model: Option<&str>,
    ) -> Result<Vec<crate::context::RetrievedChunk>> {
        let rerank_top_k = self.config.rerank_top_k;

        if !self.config.enable_rerank || chunks.is_empty() {
            return Ok(chunks);
        }
        if strategy == Some("off") {
            return Ok(chunks);
        }

        let reranker: Arc<dyn edgequake_llm::Reranker> = match strategy {
            Some("semantic") => match self.semantic_reranker_for_model(model).await {
                Some(r) => r,
                // Semantic requested but unconfigured → fall back to BM25.
                None => match self.reranker.as_ref() {
                    Some(r) => Arc::clone(r),
                    None => return Ok(chunks),
                },
            },
            _ => match self.reranker.as_ref() {
                Some(r) => Arc::clone(r),
                None => return Ok(chunks),
            },
        };

        let documents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

        let results = reranker.rerank(query, &documents, Some(rerank_top_k)).await?;

        let score_map: std::collections::HashMap<usize, f64> =
            results.iter().map(|r| (r.index, r.relevance_score)).collect();

        let min = self.config.min_rerank_score as f64;
        let mut reranked: Vec<_> = chunks
            .iter()
            .enumerate()
            .filter_map(|(idx, chunk)| {
                score_map.get(&idx).and_then(|&score| {
                    if score >= min {
                        let mut c = chunk.clone();
                        c.score = score as f32;
                        Some(c)
                    } else {
                        None
                    }
                })
            })
            .collect();

        // Recall-preserving fallback: if every chunk scored below the
        // floor (e.g. all chunks were found via the entity-graph path and
        // contain no literal query tokens for BM25), keep the original
        // ordering. This is not an error — it's a quality signal.
        if reranked.is_empty() && !chunks.is_empty() {
            tracing::warn!(
                query = %query,
                original_chunks = chunks.len(),
                min_rerank_score = self.config.min_rerank_score,
                "All chunks filtered by reranking; falling back to original"
            );
            chunks.truncate(rerank_top_k);
            return Ok(chunks);
        }

        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked.truncate(rerank_top_k);
        Ok(reranked)
    }
}
