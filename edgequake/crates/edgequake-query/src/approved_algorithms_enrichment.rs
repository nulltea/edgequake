//! Post-retrieval enrichment: attach approved algorithms to a query context.
//!
//! Mirrors `reference_code_enrichment` but targets the workspace vector
//! store (where algorithm embeddings live tagged with `metadata.type =
//! "algorithm"`) instead of the dedicated code_artifact_embeddings table.
//!
//! Silent no-op when the storage is `None` — the app-level decision to
//! enable this feature is made by wiring up (or not) an
//! `AlgorithmVectorStorage` on the engine at boot time.

use std::sync::Arc;

use edgequake_storage::traits::{AlgorithmVectorStorage, VectorStorage};

use crate::context::{ApprovedAlgorithmSnippet, ApprovedAlgorithmStep, QueryContext};
use crate::engine::QueryRequest;

/// Default cap on approved algorithms attached per query. Each snippet is
/// small-ish (name + description + a few steps + pseudocode) so 3 is a
/// reasonable ceiling for most prompts — tune later.
const DEFAULT_MAX_ALGORITHMS: i64 = 3;
/// Cosine distance threshold. `d = 1 - cos(a,b)` so small = similar.
/// `0.6` (cos ≥ 0.4) is permissive enough to catch NL-query-to-structured-
/// algorithm matches on Qwen3-embedding (empirically 0.3–0.5) while still
/// rejecting unrelated-algorithm noise (typically ≥ 0.6).
const DEFAULT_MAX_DISTANCE: f64 = 0.6;

/// Enrich `context` with reviewer-approved algorithms matching the query.
/// Best-effort — errors are logged and swallowed so answer generation
/// always proceeds.
pub async fn enrich_with_approved_algorithms(
    context: &mut QueryContext,
    request: &QueryRequest,
    query_embedding: &[f32],
    workspace_vectors: &Arc<dyn VectorStorage>,
    storage: Option<&Arc<dyn AlgorithmVectorStorage>>,
) {
    let Some(storage) = storage else {
        return; // Feature disabled.
    };
    if let Err(e) = try_enrich(
        context,
        request,
        query_embedding,
        workspace_vectors,
        storage,
    )
    .await
    {
        tracing::warn!(error = %e, "approved-algorithms enrichment skipped");
    }
}

async fn try_enrich(
    context: &mut QueryContext,
    request: &QueryRequest,
    query_embedding: &[f32],
    workspace_vectors: &Arc<dyn VectorStorage>,
    storage: &Arc<dyn AlgorithmVectorStorage>,
) -> Result<(), EnrichmentError> {
    let tenant_id = request.tenant_id().ok_or(EnrichmentError::MissingTenant)?;
    let workspace_id = request
        .workspace_id()
        .ok_or(EnrichmentError::MissingWorkspace)?;
    let tenant_uuid = uuid::Uuid::parse_str(&tenant_id)
        .map_err(|e| EnrichmentError::Other(format!("invalid tenant_id: {e}")))?;
    let workspace_uuid = uuid::Uuid::parse_str(&workspace_id)
        .map_err(|e| EnrichmentError::Other(format!("invalid workspace_id: {e}")))?;

    if request.query.trim().is_empty() || query_embedding.is_empty() {
        return Ok(());
    }

    let max_algorithms: i64 = std::env::var("EDGEQUAKE_QUERY_APPROVED_ALGORITHMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_ALGORITHMS);
    let max_distance: f64 = std::env::var("EDGEQUAKE_QUERY_ALGORITHM_MAX_DISTANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_DISTANCE);

    // Scope to documents already surfaced by main retrieval, so a query
    // about paper A can't pull in an approved algorithm from paper B just
    // because its name happens to match. Empty set → skip.
    let document_ids = collect_retrieved_document_ids(context);
    if document_ids.is_empty() {
        return Ok(());
    }

    let hits = storage
        .search_approved_algorithms(
            tenant_uuid,
            workspace_uuid,
            workspace_vectors,
            query_embedding,
            max_algorithms,
            max_distance,
            Some(&document_ids),
        )
        .await
        .map_err(|e| EnrichmentError::Storage(e.to_string()))?;

    if hits.is_empty() {
        tracing::debug!("no approved algorithms within distance threshold for query");
        return Ok(());
    }

    let top_distance = hits.first().map(|h| h.cosine_distance).unwrap_or(f64::NAN);
    context.approved_algorithms.reserve(hits.len());
    for h in hits {
        context.approved_algorithms.push(ApprovedAlgorithmSnippet {
            algorithm_id: h.algorithm_id,
            document_id: h.document_id,
            name: h.name,
            algorithm_type: h.algorithm_type,
            description: h.description,
            pseudocode: h.pseudocode,
            complexity: h.complexity,
            steps: h
                .steps
                .into_iter()
                .map(|s| ApprovedAlgorithmStep {
                    number: s.number,
                    action: s.action,
                    details: s.details,
                })
                .collect(),
            tags: h.tags,
            confidence: h.confidence,
        });
    }
    tracing::info!(
        algorithms = context.approved_algorithms.len(),
        top_distance,
        "attached approved-algorithm snippets via vector search"
    );
    Ok(())
}

fn collect_retrieved_document_ids(context: &QueryContext) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    for chunk in &context.chunks {
        if let Some(id) = chunk.document_id.as_ref().filter(|s| !s.is_empty()) {
            seen.insert(id.clone());
        }
    }
    for entity in &context.entities {
        if let Some(id) = entity.source_document_id.as_ref().filter(|s| !s.is_empty()) {
            seen.insert(id.clone());
        }
    }
    for rel in &context.relationships {
        if let Some(id) = rel.source_document_id.as_ref().filter(|s| !s.is_empty()) {
            seen.insert(id.clone());
        }
    }
    seen.into_iter().collect()
}

#[derive(Debug, thiserror::Error)]
enum EnrichmentError {
    #[error("request missing tenant_id")]
    MissingTenant,
    #[error("request missing workspace_id")]
    MissingWorkspace,
    #[error("algorithm vector storage error: {0}")]
    Storage(String),
    #[error("{0}")]
    Other(String),
}
