//! Reference-repository endpoints (Phase 0 of the Reference Code GraphRAG extension).
//!
//! Lists candidate repositories detected for a document, lets users
//! approve/reject them, and lets them manually (re-)trigger detection.

pub mod types;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use tracing::info;
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

pub use types::*;

pub fn repo_routes() -> Router<AppState> {
    Router::new()
        .route("/by-document/{document_id}", get(list_repos))
        .route("/{repo_id}/review", post(review_repo))
        .route("/detect/{document_id}", post(detect_repos))
}

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

/// List detected reference-repo candidates for a document + the detection-run state.
pub async fn list_repos(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
) -> ApiResult<Json<RepoListResponse>> {
    #[cfg(feature = "postgres")]
    {
        list_repos_impl(state, tenant_ctx, document_id).await
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, document_id);
        Err(ApiError::Internal(
            "Reference-repo listing requires the postgres feature".to_string(),
        ))
    }
}

/// Approve or reject a candidate.
pub async fn review_repo(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(repo_id): Path<Uuid>,
    Json(request): Json<ReviewRepoRequest>,
) -> ApiResult<Json<RepoReviewResponse>> {
    #[cfg(feature = "postgres")]
    {
        review_repo_impl(state, tenant_ctx, repo_id, request).await
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, repo_id, request);
        Err(ApiError::Internal(
            "Reference-repo review requires the postgres feature".to_string(),
        ))
    }
}

/// Manually trigger repo detection (enqueues a `TaskType::RepoDetection` task).
pub async fn detect_repos(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
    Json(request): Json<DetectReposRequest>,
) -> ApiResult<(StatusCode, Json<DetectReposResponse>)> {
    #[cfg(feature = "postgres")]
    {
        detect_repos_impl(state, tenant_ctx, document_id, request).await
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, document_id, request);
        Err(ApiError::Internal(
            "Reference-repo detection requires the postgres feature".to_string(),
        ))
    }
}

// ── Implementations (postgres only) ─────────────────────────────────────────

#[cfg(feature = "postgres")]
async fn list_repos_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    document_id: String,
) -> ApiResult<Json<RepoListResponse>> {
    use edgequake_agents::repo_detection::{PostgresRepoStorage, RepoStorage};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Reference-repo storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresRepoStorage::new(pool.clone());

    let rows = storage
        .list_for_document(tenant_id, workspace_id, &document_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list candidates: {e}")))?;
    let detection_run = storage
        .get_detection_run(tenant_id, workspace_id, &document_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to load detection run: {e}")))?;

    Ok(Json(RepoListResponse {
        document_id,
        candidates: rows.into_iter().map(Into::into).collect(),
        detection_run: detection_run.map(Into::into),
    }))
}

#[cfg(feature = "postgres")]
async fn review_repo_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    repo_id: Uuid,
    request: ReviewRepoRequest,
) -> ApiResult<Json<RepoReviewResponse>> {
    use edgequake_agents::repo_detection::{PostgresRepoStorage, RepoStorage};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Reference-repo storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresRepoStorage::new(pool.clone());

    let updated = storage
        .update_status(repo_id, tenant_id, workspace_id, request.status)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to update repo status: {e}")))?;

    if !updated {
        return Err(ApiError::NotFound(format!("repo {repo_id} not found")));
    }

    Ok(Json(RepoReviewResponse {
        id: repo_id,
        status: types::status_str(request.status),
    }))
}

#[cfg(feature = "postgres")]
async fn detect_repos_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    document_id: String,
    request: DetectReposRequest,
) -> ApiResult<(StatusCode, Json<DetectReposResponse>)> {
    use edgequake_tasks::{RepoDetectionData, Task, TaskType};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;

    let task_data = RepoDetectionData {
        document_id: document_id.clone(),
        workspace_id: workspace_id.to_string(),
        pdf_id: request.pdf_id,
    };

    let task = Task::new(
        tenant_id,
        workspace_id,
        TaskType::RepoDetection,
        serde_json::to_value(task_data)
            .map_err(|e| ApiError::Internal(format!("Failed to serialize task data: {e}")))?,
    );
    let track_id = task.track_id.clone();

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
        document_id = %document_id,
        track_id = %track_id,
        "Reference-repo detection task queued"
    );

    Ok((
        StatusCode::ACCEPTED,
        Json(DetectReposResponse {
            document_id,
            track_id,
            status: "queued",
        }),
    ))
}
