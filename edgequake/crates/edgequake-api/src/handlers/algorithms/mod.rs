//! Algorithm extraction and retrieval endpoints.
//!
//! Self-contained extension for extracting structured algorithm definitions from documents
//! using a 3-pass LLM pipeline (inventory → extraction → verification).

pub mod types;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use tracing::info;
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

use edgequake_algorithms::AlgorithmStatus;

pub use types::*;

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

    let limit = params.limit.unwrap_or(20).min(100);
    let offset = params.offset.unwrap_or(0);

    let (algorithms, total) = storage
        .search_algorithms(
            tenant_id,
            workspace_id,
            params.query.as_deref(),
            status,
            params.document_id.as_deref(),
            limit,
            offset,
        )
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to search algorithms: {e}")))?;

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
