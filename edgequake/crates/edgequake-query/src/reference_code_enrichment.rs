//! Post-retrieval enrichment: attach approved reference-code snippets to a
//! query context (Phase 1 of the Reference Code GraphRAG extension).
//!
//! Called by the query engine after `balance_context`. Does **not** know
//! about Postgres or any specific storage backend — it receives a
//! [`CodeVectorStorage`] trait object and a [`JinaEmbedder`] from the
//! engine and calls them directly.
//!
//! Silent no-op when either the storage or the embedder is `None` — wiring
//! both up is the app-level decision of whether this feature is enabled.

use edgequake_agents::code_analysis::JinaEmbedder;
use edgequake_storage::traits::CodeVectorStorage;
use std::sync::Arc;

use crate::context::{QueryContext, ReferenceCodeSnippet};
use crate::engine::QueryRequest;

/// Max approved snippets we'll inline. Snippets are bounded to ~8 KB each
/// by the analyzer, so 5 × 8 KB = 40 KB is a safe upper bound.
const DEFAULT_MAX_SNIPPETS: i64 = 5;
/// Cosine distance threshold. `d = 1 - cos(a,b)` so small = similar. 0.6
/// ≈ cos ≥ 0.4, permissive for now; tune as we gather ground truth.
const DEFAULT_MAX_DISTANCE: f64 = 0.6;

/// Enrich `context` with approved reference-code snippets via semantic
/// vector search. Best-effort — errors are logged and swallowed so answer
/// generation always proceeds.
pub async fn enrich_with_reference_code(
    context: &mut QueryContext,
    request: &QueryRequest,
    storage: Option<&Arc<dyn CodeVectorStorage>>,
    embedder: Option<&Arc<JinaEmbedder>>,
) {
    let (Some(storage), Some(embedder)) = (storage, embedder) else {
        return; // Feature disabled.
    };
    if let Err(e) = try_enrich(context, request, storage, embedder).await {
        tracing::warn!(error = %e, "reference-code enrichment skipped");
    }
}

async fn try_enrich(
    context: &mut QueryContext,
    request: &QueryRequest,
    storage: &Arc<dyn CodeVectorStorage>,
    embedder: &Arc<JinaEmbedder>,
) -> Result<(), EnrichmentError> {
    let tenant_id = request.tenant_id().ok_or(EnrichmentError::MissingTenant)?;
    let workspace_id = request
        .workspace_id()
        .ok_or(EnrichmentError::MissingWorkspace)?;
    let tenant_uuid = uuid::Uuid::parse_str(&tenant_id)
        .map_err(|e| EnrichmentError::Other(format!("invalid tenant_id: {e}")))?;
    let workspace_uuid = uuid::Uuid::parse_str(&workspace_id)
        .map_err(|e| EnrichmentError::Other(format!("invalid workspace_id: {e}")))?;

    if request.query.trim().is_empty() {
        return Ok(());
    }

    let max_snippets: i64 = std::env::var("EDGEQUAKE_QUERY_CODE_SNIPPETS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_SNIPPETS);
    let max_distance: f64 = std::env::var("EDGEQUAKE_QUERY_CODE_MAX_DISTANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_DISTANCE);

    let query_vec = embedder
        .embed_query_for_code_search(&request.query)
        .await
        .map_err(|e| EnrichmentError::Embedder(e.to_string()))?;

    // Scope the vector search to documents surfaced by the main retrieval.
    // Prevents cross-paper contamination in workspaces with >1 approved
    // code_artifact: the code for algorithm X from paper A shouldn't bleed
    // into answers about paper B's similarly-named algorithm.
    // Empty set → skip enrichment entirely (main retrieval hit no docs).
    let document_ids = collect_retrieved_document_ids(context);
    if document_ids.is_empty() {
        return Ok(());
    }

    let hits = storage
        .search_approved_code(
            tenant_uuid,
            workspace_uuid,
            &query_vec,
            max_snippets,
            max_distance,
            Some(&document_ids),
        )
        .await
        .map_err(|e| EnrichmentError::Storage(e.to_string()))?;

    if hits.is_empty() {
        tracing::debug!("no code embeddings within distance threshold for query");
        return Ok(());
    }

    let top_distance = hits.first().map(|h| h.cosine_distance).unwrap_or(f64::NAN);
    context.reference_code.reserve(hits.len());
    for h in hits {
        context.reference_code.push(ReferenceCodeSnippet {
            algorithm_id: h.algorithm_id,
            algorithm_name: h.algorithm_name,
            document_id: h.document_id,
            file_path: h.file_path,
            start_line: h.start_line,
            end_line: h.end_line,
            language: h.language,
            snippet: h.snippet,
            repo_url: h.repo_url,
            repo_commit: h.repo_commit,
            match_rationale: h.match_rationale,
        });
    }
    tracing::info!(
        snippets = context.reference_code.len(),
        top_distance,
        "attached reference-code snippets via vector search"
    );
    Ok(())
}

/// Collect distinct `document_id`s that main retrieval surfaced in the
/// context (chunks + entities + relationships). These define the "paper
/// allow-list" the code search is restricted to.
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
    #[error("code embedder error: {0}")]
    Embedder(String),
    #[error("code vector storage error: {0}")]
    Storage(String),
    #[error("{0}")]
    Other(String),
}
