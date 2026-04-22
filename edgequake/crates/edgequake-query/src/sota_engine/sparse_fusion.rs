use std::collections::HashMap;

use crate::context::{QueryContext, RetrievedChunk};
use crate::error::Result;
use crate::helpers::build_chunk_from_sparse_result;

use edgequake_storage::traits::MetadataFilter;

use super::SOTAQueryEngine;

impl SOTAQueryEngine {
    pub(super) async fn query_sparse_chunks(
        &self,
        query: &str,
        tenant_id: Option<String>,
        workspace_id: Option<String>,
    ) -> Result<QueryContext> {
        let mut context = QueryContext::new();
        if !self.config.enable_sparse_hybrid {
            return Ok(context);
        }
        let Some(storage) = self.sparse_chunk_storage() else {
            return Ok(context);
        };

        let mut filter =
            MetadataFilter::from_tenant_workspace(tenant_id, workspace_id).unwrap_or_default();
        filter.vector_type = Some("chunk".to_string());

        let results = match storage
            .search_chunks_bm25(query, self.config.sparse_top_k, Some(&filter))
            .await
        {
            Ok(results) => results,
            Err(error) => {
                tracing::warn!(%error, "Sparse BM25 retrieval failed; continuing without sparse chunks");
                return Ok(context);
            }
        };

        for result in results {
            context.add_chunk(build_chunk_from_sparse_result(&result));
        }

        Ok(context)
    }

    pub(super) fn rrf_fuse_chunks(
        &self,
        ranked_lists: Vec<&[RetrievedChunk]>,
    ) -> Vec<RetrievedChunk> {
        rrf_fuse_chunks_with_k(ranked_lists, self.config.max_chunks, self.config.rrf_k)
    }
}

pub(crate) fn rrf_fuse_chunks_with_k(
    ranked_lists: Vec<&[RetrievedChunk]>,
    limit: usize,
    rrf_k: f32,
) -> Vec<RetrievedChunk> {
    let mut score_map: HashMap<String, f32> = HashMap::new();
    let mut chunk_map: HashMap<String, RetrievedChunk> = HashMap::new();

    for list in ranked_lists {
        for (rank, chunk) in list.iter().enumerate() {
            let score = 1.0 / (rrf_k + (rank + 1) as f32);
            *score_map.entry(chunk.id.clone()).or_insert(0.0) += score;
            chunk_map
                .entry(chunk.id.clone())
                .and_modify(|existing| {
                    if chunk.score > existing.score {
                        *existing = chunk.clone();
                    }
                })
                .or_insert_with(|| chunk.clone());
        }
    }

    let mut fused: Vec<_> = score_map
        .into_iter()
        .filter_map(|(id, score)| {
            chunk_map.remove(&id).map(|mut chunk| {
                chunk.score = score;
                chunk
            })
        })
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    fused.truncate(limit);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RetrievedChunk;

    #[test]
    fn rrf_promotes_overlap_across_lists() {
        let dense = vec![
            RetrievedChunk::new("dense-only", "dense", 0.99),
            RetrievedChunk::new("shared", "shared dense", 0.5),
        ];
        let sparse = vec![
            RetrievedChunk::new("shared", "shared sparse", 5.0),
            RetrievedChunk::new("sparse-only", "sparse", 4.0),
        ];

        let fused = rrf_fuse_chunks_with_k(vec![&dense, &sparse], 10, 60.0);

        assert_eq!(fused[0].id, "shared");
        assert_eq!(fused.len(), 3);
    }
}
