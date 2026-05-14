//! BM25 reranking step.
//!
//! Rescores the in-memory candidate chunks against the query using the
//! configured [`Reranker`]. Boosts chunks containing rare query tokens
//! (e.g. proper nouns) that pure vector cosine would smear into the
//! surrounding semantic cluster.
//!
//! The reranker scores candidates in-memory and needs no external index
//! — see `state/mod.rs::create_bm25_reranker` for the wiring.

use super::SOTAQueryEngine;

impl SOTAQueryEngine {
    /// Re-order `chunks` by BM25 relevance to `query`.
    ///
    /// * `enable_override` — `Some(false)` skips the step regardless of
    ///   the engine config; `None` falls back to `config.enable_rerank`.
    /// * Output is truncated to `config.rerank_top_k`.
    /// * Chunks below `config.min_rerank_score` are dropped, with a
    ///   recall-preserving fallback to the original top-K if every
    ///   candidate falls under the floor.
    pub(super) async fn rerank_chunks(
        &self,
        query: &str,
        mut chunks: Vec<crate::context::RetrievedChunk>,
        enable_override: Option<bool>,
    ) -> Vec<crate::context::RetrievedChunk> {
        let enable_rerank = enable_override.unwrap_or(self.config.enable_rerank);
        let rerank_top_k = self.config.rerank_top_k;

        if !enable_rerank || self.reranker.is_none() || chunks.is_empty() {
            return chunks;
        }

        let reranker = self.reranker.as_ref().unwrap();

        let documents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

        match reranker.rerank(query, &documents, Some(rerank_top_k)).await {
            Ok(results) => {
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

                // Recall-preserving fallback: if BM25 rejected everything
                // (e.g. all chunks were found via the entity-graph path and
                // contain no literal query tokens), keep the original ordering.
                if reranked.is_empty() && !chunks.is_empty() {
                    tracing::warn!(
                        query = %query,
                        original_chunks = chunks.len(),
                        min_rerank_score = self.config.min_rerank_score,
                        "All chunks filtered by reranking; falling back to original"
                    );
                    chunks.truncate(rerank_top_k);
                    return chunks;
                }

                reranked.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                reranked.truncate(rerank_top_k);
                reranked
            }
            Err(e) => {
                tracing::warn!(error = %e, "Reranking failed; returning original chunks");
                chunks.truncate(rerank_top_k);
                chunks
            }
        }
    }
}
