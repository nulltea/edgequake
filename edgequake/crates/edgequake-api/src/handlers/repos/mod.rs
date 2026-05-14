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
        .route("/by-document/{document_id}/add", post(add_repo_manual))
        .route("/{repo_id}/review", post(review_repo))
        .route("/{repo_id}/index", post(index_repo))
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

/// `POST /repos/{repo_id}/index` — kick off Phase-2 reference-codebase
/// indexing (clone + tree-sitter + code embeddings + code graph) for an
/// approved repo, **without** going through algorithm approval first.
///
/// Fills the gap when a document has a known reference repo but no algorithms:
/// the existing auto-trigger fires only on `code_artifact` approval, and the
/// explicit `POST /reference-codebase/indexes` defaults to
/// `algorithm_focused` mode which the processor refuses without at least one
/// approved artifact. This handler chooses the right mode automatically.
///
/// Modes (caller can override via the body):
/// - `"algorithm_focused"` — pinned to approved code_artifact symbols. Useful
///   when artifacts exist; required by the original auto-trigger flow.
/// - `"full"` — index every tree-sitter symbol the analyzer surfaces.
/// - Unset (default) — pick `algorithm_focused` if there's at least one
///   approved code_artifact for this repo, otherwise `"full"`.
pub async fn index_repo(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(repo_id): Path<Uuid>,
    body: Option<Json<IndexRepoRequest>>,
) -> ApiResult<(StatusCode, Json<IndexRepoResponse>)> {
    #[cfg(feature = "postgres")]
    {
        index_repo_impl(
            state,
            tenant_ctx,
            repo_id,
            body.map(|Json(r)| r).unwrap_or_default(),
        )
        .await
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, repo_id, body);
        Err(ApiError::Internal(
            "Reference-codebase indexing requires the postgres feature".to_string(),
        ))
    }
}

/// Manually add a reference-repo row for a document by pasting a URL.
/// Bypasses Layer A link-parsing and Layer B web-search.
pub async fn add_repo_manual(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
    Json(request): Json<AddRepoRequest>,
) -> ApiResult<(StatusCode, Json<RepoCandidateResponse>)> {
    #[cfg(feature = "postgres")]
    {
        use edgequake_agents::repo_detection::{PostgresRepoStorage, RepoStorage};

        let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
        let pool = state.pg_pool.as_ref().ok_or_else(|| {
            ApiError::Internal("Reference-repo storage requires PostgreSQL pool".to_string())
        })?;
        let storage = PostgresRepoStorage::new(pool.clone());

        let trimmed = request.url.trim();
        if trimmed.is_empty() {
            return Err(ApiError::BadRequest("url must not be empty".to_string()));
        }
        let row = storage
            .insert_manual(tenant_id, workspace_id, &document_id, trimmed)
            .await
            .map_err(|e| match e {
                edgequake_agents::repo_detection::RepoStorageError::Decode(msg) => {
                    ApiError::BadRequest(msg)
                }
                other => ApiError::Internal(format!("Failed to add repo: {other}")),
            })?;

        info!(%document_id, url=%trimmed, "Manually added reference repo");
        Ok((StatusCode::CREATED, Json(RepoCandidateResponse::from(row))))
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, document_id, request);
        Err(ApiError::Internal(
            "Reference-repo add requires the postgres feature".to_string(),
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
    use edgequake_agents::repo_detection::{PostgresRepoStorage, RepoStatus, RepoStorage};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Reference-repo storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresRepoStorage::new(pool.clone());

    // WHY delete on reject: a "rejected" row has no downstream use and
    // clutters the References tab on every list. Re-detection or manual
    // add brings the repo back if the user changes their mind.
    // `code_artifacts.document_repo_id` has ON DELETE CASCADE, so any
    // analyzer output tied to this repo goes with it.
    if matches!(request.status, RepoStatus::Rejected) {
        let deleted = storage
            .delete(repo_id, tenant_id, workspace_id)
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to delete repo: {e}")))?;
        if !deleted {
            return Err(ApiError::NotFound(format!("repo {repo_id} not found")));
        }
        return Ok(Json(RepoReviewResponse {
            id: repo_id,
            status: types::status_str(RepoStatus::Rejected),
        }));
    }

    let updated = storage
        .update_status(repo_id, tenant_id, workspace_id, request.status)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to update repo status: {e}")))?;

    if !updated {
        return Err(ApiError::NotFound(format!("repo {repo_id} not found")));
    }

    // Auto-trigger Phase 1 analysis when a repo is newly approved. Best-effort:
    // if enqueue fails, the review still succeeds — the user can retry via the
    // explicit `POST /api/v1/code-reference/analyze/{repo_id}` endpoint.
    if matches!(
        request.status,
        edgequake_agents::repo_detection::RepoStatus::Approved
    ) {
        if let Err(e) =
            trigger_code_reference_analysis(&state, tenant_id, workspace_id, repo_id).await
        {
            tracing::warn!(
                error = %e,
                document_repo_id = %repo_id,
                "auto-trigger of code-reference analysis failed (non-fatal)"
            );
        }
    }

    Ok(Json(RepoReviewResponse {
        id: repo_id,
        status: types::status_str(request.status),
    }))
}

/// Look up the repo's document_id and enqueue a `CodeReferenceAnalysis` task.
#[cfg(feature = "postgres")]
async fn trigger_code_reference_analysis(
    state: &AppState,
    tenant_id: Uuid,
    workspace_id: Uuid,
    document_repo_id: Uuid,
) -> Result<(), String> {
    use edgequake_tasks::{CodeReferenceAnalysisData, Task, TaskType};

    let pool = state
        .pg_pool
        .as_ref()
        .ok_or_else(|| "pg pool unavailable".to_string())?;
    let document_id: Option<String> = sqlx::query_scalar(
        r#"SELECT document_id FROM document_repos
           WHERE id = $1 AND tenant_id = $2 AND workspace_id = $3"#,
    )
    .bind(document_repo_id)
    .bind(tenant_id)
    .bind(workspace_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("look up document_repo: {e}"))?;
    let Some(document_id) = document_id else {
        return Err("document_repo missing after status flip".into());
    };
    let task_data = CodeReferenceAnalysisData {
        document_id,
        workspace_id: workspace_id.to_string(),
        document_repo_id,
    };
    let task = Task::new(
        tenant_id,
        workspace_id,
        TaskType::CodeReferenceAnalysis,
        serde_json::to_value(&task_data).map_err(|e| format!("serialize task data: {e}"))?,
    );
    state
        .task_storage
        .create_task(&task)
        .await
        .map_err(|e| format!("create task: {e}"))?;
    state
        .task_queue
        .send(task)
        .await
        .map_err(|e| format!("queue task: {e}"))?;
    tracing::info!(
        %document_repo_id,
        "auto-triggered CodeReferenceAnalysis after repo approval"
    );
    Ok(())
}

#[cfg(feature = "postgres")]
async fn index_repo_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    repo_id: Uuid,
    request: IndexRepoRequest,
) -> ApiResult<(StatusCode, Json<IndexRepoResponse>)> {
    use edgequake_tasks::{ReferenceCodebaseIndexData, Task, TaskType};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Reference-codebase indexing requires PostgreSQL pool".to_string())
    })?;

    // Resolve the repo. Must exist, belong to caller's tenant+workspace, and
    // be approved — the indexer rejects non-approved repos anyway, but we
    // surface a 400 here so the UI gets a clean error instead of a queued-
    // then-failed task.
    let row: Option<(String, String)> = sqlx::query_as(
        r#"
        SELECT document_id, status
        FROM document_repos
        WHERE id = $1 AND tenant_id = $2 AND workspace_id = $3
        "#,
    )
    .bind(repo_id)
    .bind(tenant_id)
    .bind(workspace_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| ApiError::Internal(format!("look up document_repo: {e}")))?;

    let Some((document_id, status)) = row else {
        return Err(ApiError::NotFound(format!("repo {repo_id} not found")));
    };
    if status != "approved" {
        return Err(ApiError::BadRequest(format!(
            "repo {repo_id} is {status} — approve it first via POST /repos/{repo_id}/review"
        )));
    }

    // Auto-pick mode when the caller didn't specify. `algorithm_focused`
    // requires at least one approved code_artifact pointing at this repo; if
    // none exist (no algorithms extracted, or none survived approval) the
    // processor would refuse the task with a non-actionable error. Falling
    // back to `full` indexes every tree-sitter symbol the analyzer returns —
    // heavier, but the only correct option in this state.
    let mode = match request.mode.as_deref() {
        Some(m) if m == "algorithm_focused" || m == "full" => m.to_string(),
        Some(other) => {
            return Err(ApiError::BadRequest(format!(
                "mode must be 'algorithm_focused' or 'full', got {other:?}"
            )));
        }
        None => {
            let artifact_count: i64 = sqlx::query_scalar(
                r#"
                SELECT COUNT(*)
                FROM code_artifacts
                WHERE tenant_id = $1
                  AND workspace_id = $2
                  AND document_repo_id = $3
                  AND status = 'approved'
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(repo_id)
            .fetch_one(pool)
            .await
            .map_err(|e| ApiError::Internal(format!("count code_artifacts: {e}")))?;
            if artifact_count > 0 {
                "algorithm_focused".to_string()
            } else {
                "full".to_string()
            }
        }
    };

    let mode_static: &'static str = if mode == "algorithm_focused" {
        "algorithm_focused"
    } else {
        "full"
    };

    let task_data = ReferenceCodebaseIndexData {
        document_id: document_id.clone(),
        workspace_id: workspace_id.to_string(),
        document_repo_id: repo_id,
        mode,
        force_reindex: request.force_reindex,
    };
    let task = Task::new(
        tenant_id,
        workspace_id,
        TaskType::ReferenceCodebaseIndex,
        serde_json::to_value(&task_data)
            .map_err(|e| ApiError::Internal(format!("serialize task data: {e}")))?,
    );
    let track_id = task.track_id.clone();
    state
        .task_storage
        .create_task(&task)
        .await
        .map_err(|e| ApiError::Internal(format!("create task: {e}")))?;
    state
        .task_queue
        .send(task)
        .await
        .map_err(|e| ApiError::Internal(format!("queue task: {e}")))?;

    info!(
        %repo_id,
        %document_id,
        %mode_static,
        force_reindex = request.force_reindex,
        "queued reference-codebase index task via POST /repos/{repo_id}/index"
    );

    Ok((
        StatusCode::ACCEPTED,
        Json(IndexRepoResponse {
            repo_id,
            document_id,
            mode: mode_static,
            track_id,
            status: "queued",
        }),
    ))
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

    // Resolve pdf_id from pdf_documents when the caller didn't supply one.
    // Without this, the frontend's `detectRepos(documentId)` call landed
    // here with `pdf_id=None`, the orchestrator skipped Layer A (PDF-link
    // parsing) entirely, and re-detect just ran Layer B against markdown
    // — which typically returns nothing new. The fix keeps a user-supplied
    // override working and only auto-fills when absent.
    let pdf_id = match request.pdf_id {
        Some(id) => Some(id),
        None => {
            if let Some(pool) = state.pg_pool.as_ref() {
                let doc_uuid = uuid::Uuid::parse_str(&document_id).ok();
                if let Some(doc_uuid) = doc_uuid {
                    sqlx::query_scalar::<_, uuid::Uuid>(
                        r#"SELECT pdf_id FROM public.pdf_documents
                           WHERE document_id = $1 AND workspace_id = $2"#,
                    )
                    .bind(doc_uuid)
                    .bind(workspace_id)
                    .fetch_optional(pool)
                    .await
                    .map_err(|e| ApiError::Internal(format!("pdf_id lookup: {e}")))?
                    .map(|u| u.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        }
    };

    let task_data = RepoDetectionData {
        document_id: document_id.clone(),
        workspace_id: workspace_id.to_string(),
        pdf_id,
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
