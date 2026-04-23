//! Algorithm extraction and retrieval endpoints.
//!
//! Self-contained extension for extracting structured algorithm definitions from documents
//! using a 3-pass LLM pipeline (inventory → extraction → verification).

pub mod types;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use std::collections::{HashMap, HashSet};
use tracing::{info, warn};
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

use edgequake_algorithms::AlgorithmStatus;
use edgequake_storage::traits::{MetadataFilter, VectorStorage};

pub use types::*;

#[cfg(feature = "postgres")]
const DOCUMENT_NOT_RESOLVED_MESSAGE: &str =
    "document is not resolved, use document_list(search=...) with paper title, author name, or scheme/protocol name";

/// Create the algorithm routes sub-router.
pub fn algorithm_routes() -> Router<AppState> {
    Router::new()
        .route("/extract", post(extract_algorithms))
        .route("/by-document/{document_id}", get(list_algorithms))
        .route("/by-document/{document_id}", delete(delete_algorithms))
        .route("/by-document/{document_id}/submit", post(submit_algorithms))
        .route("/search", get(search_algorithms))
        .route("/counts", get(algorithm_counts))
        .route("/{algorithm_id}/review", post(review_algorithm))
        .route("/{algorithm_id}", delete(delete_single_algorithm))
}

/// Parse tenant/workspace UUIDs from TenantContext.
#[cfg(feature = "postgres")]
fn parse_tenant_context(ctx: &TenantContext) -> ApiResult<(Uuid, Uuid)> {
    let tenant_id_str = ctx
        .tenant_id
        .as_ref()
        .ok_or_else(|| ApiError::BadRequest("tenant_id is required".to_string()))?;
    let tenant_id = Uuid::parse_str(tenant_id_str)
        .map_err(|e| ApiError::BadRequest(format!("Invalid tenant_id: {e}")))?;
    let workspace_id_str = ctx
        .workspace_id
        .as_ref()
        .ok_or_else(|| ApiError::BadRequest("workspace_id is required".to_string()))?;
    let workspace_id = Uuid::parse_str(workspace_id_str)
        .map_err(|e| ApiError::BadRequest(format!("Invalid workspace_id: {e}")))?;
    Ok((tenant_id, workspace_id))
}

#[cfg(feature = "postgres")]
fn normalize_search_query(query: Option<&str>) -> Option<String> {
    query
        .map(str::trim)
        .filter(|q| !q.is_empty())
        .map(ToOwned::to_owned)
}

#[cfg(feature = "postgres")]
fn normalize_match_text(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
}

#[cfg(feature = "postgres")]
fn document_id_from_source_id(source_id: &str) -> Option<String> {
    let candidate = source_id.split("-chunk-").next().unwrap_or(source_id);
    Uuid::parse_str(candidate).ok()?;
    Some(candidate.to_string())
}

#[cfg(feature = "postgres")]
fn document_id_from_vector_result(
    result: &edgequake_storage::traits::VectorSearchResult,
) -> Option<String> {
    result
        .metadata
        .get("document_id")
        .or_else(|| result.metadata.get("source_document_id"))
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
        .or_else(|| edgequake_query::helpers::extract_document_id(&result.id))
}

#[cfg(feature = "postgres")]
fn graph_source_document_scores(
    nodes: Vec<(edgequake_storage::traits::GraphNode, usize)>,
) -> HashMap<String, f64> {
    let mut scores = HashMap::new();

    for (node, degree) in nodes {
        let node_score = 1.0 + (degree as f64).ln_1p();

        if let Some(source_ids) = node.properties.get("source_ids").and_then(|v| v.as_array()) {
            for source_id in source_ids.iter().filter_map(|v| v.as_str()) {
                if let Some(document_id) = document_id_from_source_id(source_id) {
                    *scores.entry(document_id).or_insert(0.0) += node_score;
                }
            }
        }

        if let Some(source_id) = node.properties.get("source_id").and_then(|v| v.as_str()) {
            for source_part in source_id.split('|') {
                if let Some(document_id) = document_id_from_source_id(source_part) {
                    *scores.entry(document_id).or_insert(0.0) += node_score;
                }
            }
        }
    }

    scores
}

#[cfg(feature = "postgres")]
#[derive(Debug, Clone)]
struct DocumentMetadataCandidate {
    document_id: String,
    title: String,
}

#[cfg(feature = "postgres")]
#[derive(Debug, Clone, Default)]
struct DocumentFocusCandidate {
    document_id: String,
    dense_score: f64,
    graph_score: f64,
    entity_score: f64,
    title_score: f64,
}

#[cfg(feature = "postgres")]
impl DocumentFocusCandidate {
    fn signal_count(&self) -> usize {
        usize::from(self.dense_score > 0.0)
            + usize::from(self.graph_score > 0.0)
            + usize::from(self.entity_score > 0.0)
            + usize::from(self.title_score > 0.0)
    }

    fn final_score(&self, max_dense: f64, max_graph: f64, max_entity: f64, max_title: f64) -> f64 {
        let dense = if max_dense > 0.0 {
            self.dense_score / max_dense
        } else {
            0.0
        };
        let graph = if max_graph > 0.0 {
            self.graph_score / max_graph
        } else {
            0.0
        };
        let entity = if max_entity > 0.0 {
            self.entity_score / max_entity
        } else {
            0.0
        };
        let title = if max_title > 0.0 {
            self.title_score / max_title
        } else {
            0.0
        };

        (0.50 * entity) + (0.20 * dense) + (0.20 * graph) + (0.10 * title)
    }
}

#[cfg(feature = "postgres")]
fn select_focus_document(mut candidates: Vec<DocumentFocusCandidate>) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }

    let max_dense = candidates
        .iter()
        .map(|candidate| candidate.dense_score)
        .fold(0.0, f64::max);
    let max_graph = candidates
        .iter()
        .map(|candidate| candidate.graph_score)
        .fold(0.0, f64::max);
    let max_entity = candidates
        .iter()
        .map(|candidate| candidate.entity_score)
        .fold(0.0, f64::max);
    let max_title = candidates
        .iter()
        .map(|candidate| candidate.title_score)
        .fold(0.0, f64::max);

    candidates.sort_by(|a, b| {
        b.final_score(max_dense, max_graph, max_entity, max_title)
            .partial_cmp(&a.final_score(max_dense, max_graph, max_entity, max_title))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.entity_score
                    .partial_cmp(&a.entity_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let top = &candidates[0];
    let top_score = top.final_score(max_dense, max_graph, max_entity, max_title);
    let second_score = candidates
        .get(1)
        .map(|candidate| candidate.final_score(max_dense, max_graph, max_entity, max_title))
        .unwrap_or(0.0);

    let clear_margin = candidates.len() == 1
        || (top_score - second_score) >= 0.12
        || top_score >= (second_score * 1.20);

    let resolved =
        top.entity_score > 0.0 && top.signal_count() >= 2 && top_score >= 0.45 && clear_margin;

    resolved.then(|| top.document_id.clone())
}

#[cfg(feature = "postgres")]
async fn load_workspace_document_metadata(
    kv_storage: &dyn edgequake_storage::traits::KVStorage,
    tenant_id: &str,
    workspace_id: &str,
) -> Result<Vec<DocumentMetadataCandidate>, ApiError> {
    let keys = kv_storage
        .keys()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list document metadata keys: {e}")))?;

    let metadata_keys: Vec<String> = keys
        .into_iter()
        .filter(|key| key.ends_with("-metadata"))
        .collect();

    if metadata_keys.is_empty() {
        return Ok(Vec::new());
    }

    let values = kv_storage
        .get_by_ids(&metadata_keys)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to fetch document metadata: {e}")))?;

    let mut documents = Vec::new();
    for value in values {
        let Some(obj) = value.as_object() else {
            continue;
        };
        let Some(document_id) = obj.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(doc_tenant_id) = obj.get("tenant_id").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(doc_workspace_id) = obj.get("workspace_id").and_then(|v| v.as_str()) else {
            continue;
        };
        if doc_tenant_id != tenant_id || doc_workspace_id != workspace_id {
            continue;
        }

        let title = obj
            .get("title")
            .or_else(|| obj.get("file_name"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        documents.push(DocumentMetadataCandidate {
            document_id: document_id.to_string(),
            title,
        });
    }

    Ok(documents)
}

#[cfg(feature = "postgres")]
fn score_document_title_match(title: &str, query: &str, tokens: &[String]) -> f64 {
    let normalized_title = normalize_match_text(title);
    let normalized_query = normalize_match_text(query);
    let mut score = 0.0;

    if !normalized_query.trim().is_empty() && normalized_title.contains(normalized_query.trim()) {
        score += 12.0;
    }

    let title_terms: HashSet<&str> = normalized_title.split_whitespace().collect();
    for token in tokens {
        if token.len() < 2 {
            continue;
        }
        if title_terms.contains(token.as_str()) {
            score += 3.0;
        } else if normalized_title.contains(token) {
            score += 1.5;
        }
    }

    score
}

#[cfg(feature = "postgres")]
fn score_document_entity_title_match(title: &str, tokens: &[String]) -> f64 {
    let normalized_title = normalize_match_text(title);
    let title_terms: HashSet<&str> = normalized_title.split_whitespace().collect();
    let mut score = 0.0;

    for token in tokens.iter().filter(|token| token.len() >= 3) {
        if title_terms.contains(token.as_str()) {
            score += 10.0;
        } else if normalized_title.contains(token) {
            score += 5.0;
        }
    }

    score
}

#[cfg(feature = "postgres")]
async fn dense_document_scores(
    query_embedding: &[f32],
    tenant_id: &str,
    workspace_id: &str,
    vector_storage: &dyn VectorStorage,
) -> Result<HashMap<String, f64>, ApiError> {
    let results = vector_storage
        .query_filtered(
            query_embedding,
            80,
            None,
            Some(&MetadataFilter {
                tenant_id: Some(tenant_id.to_string()),
                workspace_id: Some(workspace_id.to_string()),
                vector_type: Some("chunk".to_string()),
                ..Default::default()
            }),
        )
        .await
        .map_err(|e| ApiError::Internal(format!("Dense document focus search failed: {e}")))?;

    let mut scores = HashMap::new();
    for (rank, result) in results.iter().enumerate() {
        let Some(document_id) = document_id_from_vector_result(result) else {
            continue;
        };
        let rank_bonus = 1.0 / ((rank + 1) as f64);
        let score = (result.score as f64).max(0.0) + rank_bonus;
        *scores.entry(document_id).or_insert(0.0) += score;
    }

    Ok(scores)
}

#[cfg(feature = "postgres")]
fn top_graph_document_scores<'a>(
    graph_storage: &'a dyn edgequake_storage::traits::GraphStorage,
    query: &'a str,
    tokens: &'a [String],
    tenant_id: &'a str,
    workspace_id: &'a str,
) -> impl std::future::Future<Output = Result<HashMap<String, f64>, ApiError>> + 'a {
    async move {
        let mut graph_nodes = Vec::new();

        match graph_storage
            .search_nodes(query, 24, None, Some(tenant_id), Some(workspace_id))
            .await
        {
            Ok(nodes) => graph_nodes.extend(nodes),
            Err(e) => {
                warn!(error = %e, "Full-query graph entity search failed while resolving algorithm document")
            }
        }

        for token in tokens.iter().take(8) {
            match graph_storage
                .search_nodes(token, 8, None, Some(tenant_id), Some(workspace_id))
                .await
            {
                Ok(nodes) => graph_nodes.extend(nodes),
                Err(e) => {
                    warn!(query = %token, error = %e, "Token graph entity search failed while resolving algorithm document")
                }
            }
        }

        Ok(graph_source_document_scores(graph_nodes))
    }
}

#[cfg(feature = "postgres")]
fn entity_name_graph_scores<'a>(
    graph_storage: &'a dyn edgequake_storage::traits::GraphStorage,
    tokens: &'a [String],
    tenant_id: &'a str,
    workspace_id: &'a str,
) -> impl std::future::Future<Output = Result<HashMap<String, f64>, ApiError>> + 'a {
    async move {
        let mut scores = HashMap::new();

        for token in tokens.iter().filter(|token| token.len() >= 3).take(8) {
            let nodes = match graph_storage
                .search_nodes(token, 12, None, Some(tenant_id), Some(workspace_id))
                .await
            {
                Ok(nodes) => nodes,
                Err(e) => {
                    warn!(query = %token, error = %e, "Entity graph search failed while resolving algorithm document");
                    continue;
                }
            };

            for (node, degree) in nodes {
                let normalized_node_id = normalize_match_text(&node.id);
                let exact_match = normalized_node_id
                    .split_whitespace()
                    .any(|part| part == token);
                let contains_match = normalized_node_id.contains(token);
                if !exact_match && !contains_match {
                    continue;
                }

                let node_score = if exact_match { 6.0 } else { 3.0 } + (degree as f64).ln_1p();
                if let Some(source_ids) =
                    node.properties.get("source_ids").and_then(|v| v.as_array())
                {
                    for source_id in source_ids.iter().filter_map(|v| v.as_str()) {
                        if let Some(document_id) = document_id_from_source_id(source_id) {
                            *scores.entry(document_id).or_insert(0.0) += node_score;
                        }
                    }
                }

                if let Some(source_id) = node.properties.get("source_id").and_then(|v| v.as_str()) {
                    for source_part in source_id.split('|') {
                        if let Some(document_id) = document_id_from_source_id(source_part) {
                            *scores.entry(document_id).or_insert(0.0) += node_score;
                        }
                    }
                }
            }
        }

        Ok(scores)
    }
}

#[cfg(feature = "postgres")]
async fn resolve_focus_document(
    state: &AppState,
    query: &str,
    tokens: &[String],
    tenant_id: Uuid,
    workspace_id: Uuid,
    embedding_provider: &dyn edgequake_query::EmbeddingProvider,
    vector_storage: &dyn VectorStorage,
) -> Result<Option<String>, ApiError> {
    let tenant_id_str = tenant_id.to_string();
    let workspace_id_str = workspace_id.to_string();

    let query_embedding = embedding_provider
        .embed_one(query)
        .await
        .map_err(|e| ApiError::Internal(format!("Algorithm query embedding failed: {e}")))?;

    let (documents, dense_scores, graph_scores, entity_graph_scores) = tokio::join!(
        load_workspace_document_metadata(
            state.kv_storage.as_ref(),
            &tenant_id_str,
            &workspace_id_str,
        ),
        dense_document_scores(
            &query_embedding,
            &tenant_id_str,
            &workspace_id_str,
            vector_storage,
        ),
        top_graph_document_scores(
            state.graph_storage.as_ref(),
            query,
            tokens,
            &tenant_id_str,
            &workspace_id_str,
        ),
        entity_name_graph_scores(
            state.graph_storage.as_ref(),
            tokens,
            &tenant_id_str,
            &workspace_id_str,
        ),
    );

    let documents = documents?;
    let dense_scores = dense_scores?;
    let graph_scores = graph_scores?;
    let entity_graph_scores = entity_graph_scores?;

    let mut candidates = HashMap::<String, DocumentFocusCandidate>::new();
    for document in documents {
        let title_score = score_document_title_match(&document.title, query, tokens)
            + score_document_entity_title_match(&document.title, tokens);
        let dense_score = dense_scores
            .get(&document.document_id)
            .copied()
            .unwrap_or(0.0);
        let graph_score = graph_scores
            .get(&document.document_id)
            .copied()
            .unwrap_or(0.0);
        let entity_score = entity_graph_scores
            .get(&document.document_id)
            .copied()
            .unwrap_or(0.0);

        if dense_score == 0.0 && graph_score == 0.0 && entity_score == 0.0 && title_score == 0.0 {
            continue;
        }

        candidates.insert(
            document.document_id.clone(),
            DocumentFocusCandidate {
                document_id: document.document_id,
                dense_score,
                graph_score,
                entity_score,
                title_score,
            },
        );
    }

    if candidates.is_empty() {
        return Ok(None);
    }

    Ok(select_focus_document(candidates.into_values().collect()))
}

#[cfg(feature = "postgres")]
fn add_algorithm_candidate(
    candidates: &mut HashMap<Uuid, AlgorithmCandidateScore>,
    algorithm: edgequake_algorithms::Algorithm,
) -> &mut AlgorithmCandidateScore {
    candidates
        .entry(algorithm.id)
        .or_insert_with(|| AlgorithmCandidateScore {
            algorithm,
            lexical_score: 0.0,
            semantic_score: 0.0,
            graph_score: 0.0,
        })
}

#[cfg(feature = "postgres")]
struct AlgorithmCandidateScore {
    algorithm: edgequake_algorithms::Algorithm,
    lexical_score: f64,
    semantic_score: f64,
    graph_score: f64,
}

#[cfg(feature = "postgres")]
impl AlgorithmCandidateScore {
    fn final_score(&self, max_lexical: f64, max_graph: f64) -> f64 {
        let lexical = if max_lexical > 0.0 {
            self.lexical_score / max_lexical
        } else {
            0.0
        };
        let graph = if max_graph > 0.0 {
            self.graph_score / max_graph
        } else {
            0.0
        };

        (0.55 * self.semantic_score) + (0.35 * lexical) + (0.10 * graph)
    }
}

/// Trigger algorithm extraction for a document.
pub async fn extract_algorithms(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Json(request): Json<ExtractAlgorithmsRequest>,
) -> ApiResult<(StatusCode, Json<ExtractAlgorithmsResponse>)> {
    extract_algorithms_impl(state, tenant_ctx, request).await
}

/// List algorithms extracted from a document.
pub async fn list_algorithms(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
    Query(params): Query<ListAlgorithmsParams>,
) -> ApiResult<Json<AlgorithmListResponse>> {
    list_algorithms_impl(state, tenant_ctx, document_id, params).await
}

/// Search algorithms across documents in a workspace.
pub async fn search_algorithms(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Query(params): Query<SearchAlgorithmsParams>,
) -> ApiResult<Json<AlgorithmSearchResponse>> {
    search_algorithms_impl(state, tenant_ctx, params).await
}

/// Approve or reject an extracted algorithm.
pub async fn review_algorithm(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(algorithm_id): Path<Uuid>,
    Json(request): Json<ReviewAlgorithmRequest>,
) -> ApiResult<Json<AlgorithmReviewResponse>> {
    review_algorithm_impl(state, tenant_ctx, algorithm_id, request).await
}

/// Delete all algorithms for a document.
pub async fn delete_algorithms(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
) -> ApiResult<Json<AlgorithmDeleteResponse>> {
    delete_algorithms_impl(state, tenant_ctx, document_id).await
}

/// Submit reviewed algorithms — queue embedding for all approved algorithms.
pub async fn submit_algorithms(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
) -> ApiResult<Json<AlgorithmSubmitResponse>> {
    submit_algorithms_impl(state, tenant_ctx, document_id).await
}

/// Delete a single algorithm by ID.
pub async fn delete_single_algorithm(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(algorithm_id): Path<Uuid>,
) -> ApiResult<Json<AlgorithmDeleteResponse>> {
    delete_single_algorithm_impl(state, tenant_ctx, algorithm_id).await
}

// ── PostgreSQL implementations ──────────────────────────────────────────────

#[cfg(feature = "postgres")]
async fn extract_algorithms_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    request: ExtractAlgorithmsRequest,
) -> ApiResult<(StatusCode, Json<ExtractAlgorithmsResponse>)> {
    use edgequake_tasks::{AlgorithmExtractionData, Task, TaskType};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;

    info!(
        document_id = %request.document_id,
        tenant_id = %tenant_id,
        workspace_id = %workspace_id,
        "Starting algorithm extraction"
    );

    // Validate document exists
    let metadata_key = format!("{}-metadata", request.document_id);
    let metadata = state
        .kv_storage
        .get_by_id(&metadata_key)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to fetch document metadata: {e}")))?
        .ok_or_else(|| {
            ApiError::NotFound(format!("Document not found: {}", request.document_id))
        })?;

    let doc_status = metadata
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    if doc_status != "completed" && doc_status != "indexed" {
        return Err(ApiError::BadRequest(format!(
            "Document must be completed before extraction (current status: {doc_status})"
        )));
    }

    // Determine source type and build task data.
    // PDF documents: pass pdf_id so the processor uses layout detection + VLM (Pass 1).
    // Text documents: fetch content and chunk it (fallback path).
    let source_type = metadata
        .get("source_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let pdf_id = metadata.get("pdf_id").and_then(|v| v.as_str());

    // Resolve vision provider/model from workspace settings for VLM recognition.
    let (vision_provider, vision_model) = {
        let ws = state
            .workspace_service
            .get_workspace(workspace_id)
            .await
            .ok()
            .flatten();
        (
            ws.as_ref().and_then(|w| w.vision_llm_provider.clone()),
            ws.as_ref().and_then(|w| w.vision_llm_model.clone()),
        )
    };

    let task_data = if source_type == "pdf" {
        // PDF path: processor will use layout detection + VLM to find algorithm blocks.
        let pdf_id_str =
            pdf_id.ok_or_else(|| ApiError::Internal("PDF document missing pdf_id".to_string()))?;

        info!(
            document_id = %request.document_id,
            pdf_id = %pdf_id_str,
            vision_provider = ?vision_provider,
            vision_model = ?vision_model,
            "Algorithm extraction: PDF document, will use layout detection + VLM"
        );

        AlgorithmExtractionData {
            document_id: request.document_id.clone(),
            workspace_id: workspace_id.to_string(),
            chunks: vec![],
            pdf_id: Some(pdf_id_str.to_string()),
            vision_provider,
            vision_model,
        }
    } else {
        // Text/markdown documents: fetch content and chunk with context-aware strategy.
        let mut stitched = String::new();
        let mut idx = 0;
        loop {
            let key = format!("{}-chunk-{}", request.document_id, idx);
            match state
                .kv_storage
                .get_by_id(&key)
                .await
                .map_err(|e| ApiError::Internal(format!("Failed to fetch chunk {idx}: {e}")))?
            {
                Some(v) => {
                    let text = v
                        .as_str()
                        .map(|s| s.to_string())
                        .or_else(|| {
                            v.get("content")
                                .and_then(|c| c.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| v.to_string());
                    if !text.is_empty() {
                        if !stitched.is_empty() {
                            stitched.push_str("\n\n");
                        }
                        stitched.push_str(&text);
                    }
                    idx += 1;
                }
                None => break,
            }
        }
        if stitched.is_empty() {
            return Err(ApiError::BadRequest(
                "Document has no content; reprocess the document first".to_string(),
            ));
        }

        let max_tokens = std::env::var("EDGEQUAKE_ALGO_CHUNK_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096);
        let min_tokens = std::env::var("EDGEQUAKE_ALGO_CHUNK_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(max_tokens / 4);

        let algo_chunker_config = edgequake_pipeline::chunker::ChunkerConfig {
            chunk_size: max_tokens,
            chunk_overlap: 0,
            min_chunk_size: min_tokens,
            ..Default::default()
        };

        let chunker_strategy = edgequake_pipeline::chunker::ContextAwareChunking;
        let chunks: Vec<String> = {
            use edgequake_pipeline::chunker::ChunkingStrategy;
            chunker_strategy
                .chunk(&stitched, &algo_chunker_config)
                .await
                .map_err(|e| ApiError::Internal(format!("Algorithm chunking failed: {e}")))?
                .into_iter()
                .map(|c| c.content)
                .collect()
        };

        if chunks.is_empty() {
            return Err(ApiError::BadRequest(
                "Context-aware chunker produced no chunks".to_string(),
            ));
        }

        info!(
            document_id = %request.document_id,
            chunk_count = chunks.len(),
            "Algorithm extraction: text document, chunked with context-aware strategy"
        );

        AlgorithmExtractionData {
            document_id: request.document_id.clone(),
            workspace_id: workspace_id.to_string(),
            chunks,
            pdf_id: None,
            vision_provider: None,
            vision_model: None,
        }
    };

    let task = Task::new(
        tenant_id,
        workspace_id,
        TaskType::AlgorithmExtraction,
        serde_json::to_value(task_data)
            .map_err(|e| ApiError::Internal(format!("Failed to serialize task data: {e}")))?,
    );
    let track_id = task.track_id.clone();

    // Store and enqueue task
    state
        .task_storage
        .create_task(&task)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to create task: {e}")))?;

    state
        .task_queue
        .send(task)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to queue task: {e}")))?;

    info!(
        document_id = %request.document_id,
        track_id = %track_id,
        "Algorithm extraction task queued"
    );

    Ok((
        StatusCode::ACCEPTED,
        Json(ExtractAlgorithmsResponse {
            document_id: request.document_id,
            status: "pending".to_string(),
            message: format!(
                "Algorithm extraction task queued. Track with task ID: {}",
                track_id
            ),
        }),
    ))
}

#[cfg(feature = "postgres")]
/// List algorithm counts grouped by document for the calling workspace.
///
/// Used by the document-list UI to decide, per-row, whether to render the
/// "Algorithms" (`</>`) action button: no row for a doc here → no button.
pub async fn algorithm_counts(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
) -> ApiResult<Json<AlgorithmCountsResponse>> {
    #[cfg(feature = "postgres")]
    {
        use edgequake_algorithms::{AlgorithmStorage, PostgresAlgorithmStorage};

        let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
        let pool = state.pg_pool.as_ref().ok_or_else(|| {
            ApiError::Internal("Algorithm storage requires PostgreSQL pool".to_string())
        })?;
        let storage = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool.clone()));

        let rows = storage
            .counts_by_document(tenant_id, workspace_id)
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to count algorithms: {e}")))?;

        let counts = rows
            .into_iter()
            .map(|(doc_id, count)| AlgorithmCountEntry {
                document_id: doc_id,
                count,
            })
            .collect();

        Ok(Json(AlgorithmCountsResponse { counts }))
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx);
        Err(ApiError::Internal(
            "Algorithm counts require the postgres feature".to_string(),
        ))
    }
}

#[cfg(feature = "postgres")]
async fn list_algorithms_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    document_id: String,
    params: ListAlgorithmsParams,
) -> ApiResult<Json<AlgorithmListResponse>> {
    use edgequake_algorithms::{AlgorithmStorage, PostgresAlgorithmStorage};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Algorithm storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool.clone()));

    let status = params
        .status
        .as_deref()
        .map(|s| {
            s.parse::<AlgorithmStatus>()
                .map_err(|e| ApiError::BadRequest(format!("Invalid status: {e}")))
        })
        .transpose()?;

    let algorithms = storage
        .list_algorithms(&document_id, tenant_id, workspace_id, status)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list algorithms: {e}")))?;

    let total = algorithms.len();
    Ok(Json(AlgorithmListResponse { algorithms, total }))
}

#[cfg(feature = "postgres")]
async fn search_algorithms_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    params: SearchAlgorithmsParams,
) -> ApiResult<Json<AlgorithmSearchResponse>> {
    use crate::handlers::query::workspace_resolve::{
        get_workspace_embedding_provider, get_workspace_vector_storage,
    };
    use edgequake_algorithms::{
        score_algorithm_match, tokenize_algorithm_query, AlgorithmStorage, PostgresAlgorithmStorage,
    };
    use edgequake_storage::{AlgorithmVectorStorage, PgAlgorithmVectorStorage};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Algorithm storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool.clone()));

    let limit = params.limit.unwrap_or(20).clamp(1, 100);
    let offset = params.offset.unwrap_or(0).max(0);
    let query = normalize_search_query(params.query.as_deref());

    if query.is_none() {
        let (algorithms, total) = storage
            .search_algorithms(
                tenant_id,
                workspace_id,
                None,
                params.document_id.as_deref(),
                limit,
                offset,
            )
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to search algorithms: {e}")))?;

        return Ok(Json(AlgorithmSearchResponse {
            algorithms,
            total,
            limit,
            offset,
        }));
    }

    let query = query.expect("query is checked above");
    let tokens = tokenize_algorithm_query(&query);
    let candidate_limit = (limit + offset).max(100).min(500);
    let tenant_id_str = tenant_id.to_string();
    let workspace_id_str = workspace_id.to_string();
    let embedding_provider = get_workspace_embedding_provider(&state, &workspace_id_str)
        .await?
        .unwrap_or_else(|| std::sync::Arc::clone(&state.embedding_provider));
    let vector_storage = get_workspace_vector_storage(&state, &workspace_id_str)
        .await?
        .unwrap_or_else(|| std::sync::Arc::clone(&state.vector_storage));
    let resolved_document_id = if let Some(document_id) = params.document_id.clone() {
        document_id
    } else {
        resolve_focus_document(
            &state,
            &query,
            &tokens,
            tenant_id,
            workspace_id,
            embedding_provider.as_ref(),
            vector_storage.as_ref(),
        )
        .await?
        .ok_or_else(|| ApiError::BadRequest(DOCUMENT_NOT_RESOLVED_MESSAGE.to_string()))?
    };
    let document_filter = vec![resolved_document_id.clone()];

    let (lexical_algorithms, _) = storage
        .search_algorithms(
            tenant_id,
            workspace_id,
            Some(&query),
            Some(&resolved_document_id),
            candidate_limit,
            0,
        )
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to search algorithms: {e}")))?;

    let mut candidates = HashMap::new();
    for algorithm in lexical_algorithms {
        let score = score_algorithm_match(&algorithm, &tokens, &query);
        add_algorithm_candidate(&mut candidates, algorithm).lexical_score = score;
    }

    match embedding_provider.embed_one(&query).await {
        Ok(query_embedding) => {
            let algorithm_vector_storage = PgAlgorithmVectorStorage::new(pool.clone());
            match algorithm_vector_storage
                .search_approved_algorithms(
                    tenant_id,
                    workspace_id,
                    &vector_storage,
                    &query_embedding,
                    candidate_limit,
                    1.0,
                    Some(document_filter.as_slice()),
                )
                .await
            {
                Ok(hits) => {
                    for hit in hits {
                        let Ok(algorithm_id) = Uuid::parse_str(&hit.algorithm_id) else {
                            continue;
                        };
                        match storage
                            .get_algorithm(algorithm_id, tenant_id, workspace_id)
                            .await
                        {
                            Ok(Some(algorithm)) => {
                                let candidate = add_algorithm_candidate(&mut candidates, algorithm);
                                candidate.semantic_score = candidate
                                    .semantic_score
                                    .max((1.0 - hit.cosine_distance).clamp(0.0, 1.0));
                            }
                            Ok(None) => {}
                            Err(e) => warn!(
                                algorithm_id = %algorithm_id,
                                error = %e,
                                "Failed to hydrate semantic algorithm search hit"
                            ),
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Semantic algorithm search failed; using lexical and graph signals")
                }
            }
        }
        Err(e) => {
            warn!(error = %e, "Algorithm query embedding failed; using lexical and graph signals")
        }
    }

    let mut graph_nodes = Vec::new();
    match state
        .graph_storage
        .search_nodes(
            &query,
            24,
            None,
            Some(&tenant_id_str),
            Some(&workspace_id_str),
        )
        .await
    {
        Ok(nodes) => graph_nodes.extend(nodes),
        Err(e) => warn!(error = %e, "Full-query graph entity search failed"),
    }
    for token in tokens.iter().take(8) {
        match state
            .graph_storage
            .search_nodes(
                token,
                8,
                None,
                Some(&tenant_id_str),
                Some(&workspace_id_str),
            )
            .await
        {
            Ok(nodes) => graph_nodes.extend(nodes),
            Err(e) => warn!(query = %token, error = %e, "Token graph entity search failed"),
        }
    }

    let graph_scores = graph_source_document_scores(graph_nodes);
    if !graph_scores.is_empty() {
        let (graph_algorithms, _) = storage
            .search_algorithms(
                tenant_id,
                workspace_id,
                None,
                Some(&resolved_document_id),
                500,
                0,
            )
            .await
            .map_err(|e| {
                ApiError::Internal(format!("Failed to load graph algorithm candidates: {e}"))
            })?;

        for algorithm in graph_algorithms {
            let Some(score) = graph_scores.get(&algorithm.document_id).copied() else {
                continue;
            };
            add_algorithm_candidate(&mut candidates, algorithm).graph_score = score;
        }
    }

    let max_lexical = candidates
        .values()
        .map(|c| c.lexical_score)
        .fold(0.0, f64::max);
    let max_graph = candidates
        .values()
        .map(|c| c.graph_score)
        .fold(0.0, f64::max);

    let mut ranked = candidates.into_values().collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.final_score(max_lexical, max_graph)
            .partial_cmp(&a.final_score(max_lexical, max_graph))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.algorithm.created_at.cmp(&a.algorithm.created_at))
    });

    let total = ranked.len() as i64;
    let algorithms = ranked
        .into_iter()
        .skip(offset as usize)
        .take(limit as usize)
        .map(|candidate| candidate.algorithm)
        .collect();

    Ok(Json(AlgorithmSearchResponse {
        algorithms,
        total,
        limit,
        offset,
    }))
}

#[cfg(feature = "postgres")]
async fn review_algorithm_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    algorithm_id: Uuid,
    request: ReviewAlgorithmRequest,
) -> ApiResult<Json<AlgorithmReviewResponse>> {
    use edgequake_algorithms::{AlgorithmStorage, PostgresAlgorithmStorage};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Algorithm storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool.clone()));

    storage
        .update_algorithm_status(algorithm_id, tenant_id, workspace_id, request.status)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to update algorithm status: {e}")))?;

    Ok(Json(AlgorithmReviewResponse {
        id: algorithm_id,
        status: request.status.to_string(),
    }))
}

#[cfg(feature = "postgres")]
async fn submit_algorithms_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    document_id: String,
) -> ApiResult<Json<AlgorithmSubmitResponse>> {
    use edgequake_algorithms::{AlgorithmStorage, PostgresAlgorithmStorage};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Algorithm storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool.clone()));

    // Fetch all algorithms for this document
    let algorithms = storage
        .list_algorithms(&document_id, tenant_id, workspace_id, None)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list algorithms: {e}")))?;

    // Verify none are pending
    let pending_count = algorithms
        .iter()
        .filter(|a| a.status == AlgorithmStatus::Pending)
        .count();
    if pending_count > 0 {
        return Err(ApiError::BadRequest(format!(
            "{pending_count} algorithm(s) still pending review. Review all before submitting."
        )));
    }

    let approved: Vec<String> = algorithms
        .iter()
        .filter(|a| a.status == AlgorithmStatus::Approved)
        .map(|a| a.id.to_string())
        .collect();
    let rejected_count = algorithms.len() - approved.len();

    if approved.is_empty() {
        return Ok(Json(AlgorithmSubmitResponse {
            document_id,
            approved_count: 0,
            rejected_count,
            status: "no_approved_algorithms".to_string(),
        }));
    }

    // Queue embedding task for approved algorithms
    queue_algorithm_embedding_task(&state, tenant_id, workspace_id, &document_id, &approved)
        .await?;

    Ok(Json(AlgorithmSubmitResponse {
        document_id,
        approved_count: approved.len(),
        rejected_count,
        status: "embedding_queued".to_string(),
    }))
}

/// Queue an algorithm embedding task for the given algorithm IDs.
#[cfg(feature = "postgres")]
async fn queue_algorithm_embedding_task(
    state: &AppState,
    tenant_id: Uuid,
    workspace_id: Uuid,
    document_id: &str,
    algorithm_ids: &[String],
) -> ApiResult<()> {
    use edgequake_tasks::{AlgorithmEmbeddingData, Task, TaskType};

    let task_data = AlgorithmEmbeddingData {
        document_id: document_id.to_string(),
        workspace_id: workspace_id.to_string(),
        algorithm_ids: algorithm_ids.to_vec(),
    };

    let task = Task::new(
        tenant_id,
        workspace_id,
        TaskType::AlgorithmEmbedding,
        serde_json::to_value(task_data)
            .map_err(|e| ApiError::Internal(format!("Failed to serialize task data: {e}")))?,
    );

    state
        .task_storage
        .create_task(&task)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to create embedding task: {e}")))?;

    state
        .task_queue
        .send(task)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to queue embedding task: {e}")))?;

    info!(document_id = %document_id, count = algorithm_ids.len(), "Algorithm embedding task queued");
    Ok(())
}

#[cfg(feature = "postgres")]
async fn delete_algorithms_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    document_id: String,
) -> ApiResult<Json<AlgorithmDeleteResponse>> {
    use edgequake_algorithms::{AlgorithmStorage, PostgresAlgorithmStorage};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Algorithm storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool.clone()));

    let deleted = storage
        .delete_algorithms_by_document(&document_id, tenant_id, workspace_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to delete algorithms: {e}")))?;

    info!(document_id = %document_id, deleted = deleted, "Deleted algorithms");
    Ok(Json(AlgorithmDeleteResponse { deleted }))
}

#[cfg(feature = "postgres")]
async fn delete_single_algorithm_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    algorithm_id: Uuid,
) -> ApiResult<Json<AlgorithmDeleteResponse>> {
    use edgequake_algorithms::{AlgorithmStorage, PostgresAlgorithmStorage};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Algorithm storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool.clone()));

    let deleted = storage
        .delete_algorithm(algorithm_id, tenant_id, workspace_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to delete algorithm: {e}")))?;

    if !deleted {
        return Err(ApiError::NotFound(format!(
            "Algorithm not found: {algorithm_id}"
        )));
    }

    info!(algorithm_id = %algorithm_id, "Deleted algorithm");
    Ok(Json(AlgorithmDeleteResponse { deleted: 1 }))
}

// ── Non-PostgreSQL fallbacks ────────────────────────────────────────────────

#[cfg(not(feature = "postgres"))]
async fn extract_algorithms_impl(
    _state: AppState,
    _tenant_ctx: TenantContext,
    _request: ExtractAlgorithmsRequest,
) -> ApiResult<(StatusCode, Json<ExtractAlgorithmsResponse>)> {
    Err(ApiError::Internal(
        "Algorithm extraction requires PostgreSQL feature".to_string(),
    ))
}

#[cfg(not(feature = "postgres"))]
async fn list_algorithms_impl(
    _state: AppState,
    _tenant_ctx: TenantContext,
    _document_id: String,
    _params: ListAlgorithmsParams,
) -> ApiResult<Json<AlgorithmListResponse>> {
    Err(ApiError::Internal(
        "Algorithms require PostgreSQL feature".to_string(),
    ))
}

#[cfg(not(feature = "postgres"))]
async fn search_algorithms_impl(
    _state: AppState,
    _tenant_ctx: TenantContext,
    _params: SearchAlgorithmsParams,
) -> ApiResult<Json<AlgorithmSearchResponse>> {
    Err(ApiError::Internal(
        "Algorithms require PostgreSQL feature".to_string(),
    ))
}

#[cfg(not(feature = "postgres"))]
async fn review_algorithm_impl(
    _state: AppState,
    _tenant_ctx: TenantContext,
    _algorithm_id: Uuid,
    _request: ReviewAlgorithmRequest,
) -> ApiResult<Json<AlgorithmReviewResponse>> {
    Err(ApiError::Internal(
        "Algorithms require PostgreSQL feature".to_string(),
    ))
}

#[cfg(not(feature = "postgres"))]
async fn delete_algorithms_impl(
    _state: AppState,
    _tenant_ctx: TenantContext,
    _document_id: String,
) -> ApiResult<Json<AlgorithmDeleteResponse>> {
    Err(ApiError::Internal(
        "Algorithms require PostgreSQL feature".to_string(),
    ))
}

#[cfg(not(feature = "postgres"))]
async fn delete_single_algorithm_impl(
    _state: AppState,
    _tenant_ctx: TenantContext,
    _algorithm_id: Uuid,
) -> ApiResult<Json<AlgorithmDeleteResponse>> {
    Err(ApiError::Internal(
        "Algorithms require PostgreSQL feature".to_string(),
    ))
}

#[cfg(not(feature = "postgres"))]
async fn submit_algorithms_impl(
    _state: AppState,
    _tenant_ctx: TenantContext,
    _document_id: String,
) -> ApiResult<Json<AlgorithmSubmitResponse>> {
    Err(ApiError::Internal(
        "Algorithms require PostgreSQL feature".to_string(),
    ))
}

#[cfg(all(test, feature = "postgres"))]
mod tests {
    use super::*;

    #[test]
    fn resolver_prefers_entity_led_caprise_document() {
        let selected = select_focus_document(vec![
            DocumentFocusCandidate {
                document_id: "sap-doc".to_string(),
                dense_score: 9.0,
                graph_score: 8.0,
                entity_score: 0.0,
                title_score: 3.0,
            },
            DocumentFocusCandidate {
                document_id: "caprise-doc".to_string(),
                dense_score: 6.0,
                graph_score: 4.0,
                entity_score: 10.0,
                title_score: 5.0,
            },
        ]);

        assert_eq!(selected.as_deref(), Some("caprise-doc"));
    }

    #[test]
    fn resolver_rejects_ambiguous_query_without_entity_anchor() {
        let selected = select_focus_document(vec![
            DocumentFocusCandidate {
                document_id: "sap-doc".to_string(),
                dense_score: 9.0,
                graph_score: 8.0,
                entity_score: 0.0,
                title_score: 4.0,
            },
            DocumentFocusCandidate {
                document_id: "caprise-doc".to_string(),
                dense_score: 8.5,
                graph_score: 7.5,
                entity_score: 0.0,
                title_score: 4.0,
            },
        ]);

        assert_eq!(selected, None);
    }
}
