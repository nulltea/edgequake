//! Reference-code analysis endpoints (Phase 1 of the Reference Code GraphRAG
//! extension).

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

pub fn code_reference_routes() -> Router<AppState> {
    Router::new()
        .route("/analyze/{document_repo_id}", post(analyze))
        .route("/by-document/{document_id}", get(list_for_document))
        .route("/{code_artifact_id}/review", post(review))
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

/// Enqueue a `CodeReferenceAnalysis` task for the given approved repo row.
pub async fn analyze(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_repo_id): Path<Uuid>,
) -> ApiResult<(StatusCode, Json<AnalyzeCodeReferenceResponse>)> {
    #[cfg(feature = "postgres")]
    {
        analyze_impl(state, tenant_ctx, document_repo_id).await
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, document_repo_id);
        Err(ApiError::Internal(
            "Code-reference analysis requires the postgres feature".to_string(),
        ))
    }
}

/// List candidate code_artifacts for a document + per-repo run state.
pub async fn list_for_document(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
) -> ApiResult<Json<CodeReferenceListResponse>> {
    #[cfg(feature = "postgres")]
    {
        list_impl(state, tenant_ctx, document_id).await
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, document_id);
        Err(ApiError::Internal(
            "Code-reference listing requires the postgres feature".to_string(),
        ))
    }
}

/// Approve / reject a candidate.
pub async fn review(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(code_artifact_id): Path<Uuid>,
    Json(request): Json<ReviewCodeArtifactRequest>,
) -> ApiResult<Json<CodeArtifactReviewResponse>> {
    #[cfg(feature = "postgres")]
    {
        review_impl(state, tenant_ctx, code_artifact_id, request).await
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, code_artifact_id, request);
        Err(ApiError::Internal(
            "Code-reference review requires the postgres feature".to_string(),
        ))
    }
}

// ── Implementations ────────────────────────────────────────────────────────

#[cfg(feature = "postgres")]
async fn analyze_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    document_repo_id: Uuid,
) -> ApiResult<(StatusCode, Json<AnalyzeCodeReferenceResponse>)> {
    use edgequake_agents::repo_detection::PostgresRepoStorage;
    use edgequake_tasks::{CodeReferenceAnalysisData, Task, TaskType};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Code-reference storage requires PostgreSQL pool".to_string())
    })?;

    // Resolve document_id via the document_repos row. We look up all rows
    // for every doc the tenant has and filter by id — small per-tenant row
    // count makes this OK for Phase 1; optimise with a direct by-id query
    // later if it becomes a hot path.
    let repo_storage = PostgresRepoStorage::new(pool.clone());
    let repo_row = find_repo_by_id(&repo_storage, tenant_id, workspace_id, document_repo_id)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("document_repo {document_repo_id} not found")))?;

    let task_data = CodeReferenceAnalysisData {
        document_id: repo_row.document_id.clone(),
        workspace_id: workspace_id.to_string(),
        document_repo_id,
    };
    let task = Task::new(
        tenant_id,
        workspace_id,
        TaskType::CodeReferenceAnalysis,
        serde_json::to_value(&task_data)
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
        document_id = %repo_row.document_id,
        document_repo_id = %document_repo_id,
        track_id = %track_id,
        "Code-reference analysis task queued"
    );
    Ok((
        StatusCode::ACCEPTED,
        Json(AnalyzeCodeReferenceResponse {
            document_id: repo_row.document_id,
            document_repo_id,
            track_id,
            status: "queued",
        }),
    ))
}

#[cfg(feature = "postgres")]
async fn list_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    document_id: String,
) -> ApiResult<Json<CodeReferenceListResponse>> {
    use edgequake_agents::code_analysis::{CodeArtifactStorage, PostgresCodeArtifactStorage};
    use edgequake_agents::repo_detection::{PostgresRepoStorage, RepoStorage};

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Code-reference storage requires PostgreSQL pool".to_string())
    })?;
    let code_storage = PostgresCodeArtifactStorage::new(pool.clone());
    let repo_storage = PostgresRepoStorage::new(pool.clone());

    let candidates = code_storage
        .list_for_document(tenant_id, workspace_id, &document_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list code_artifacts: {e}")))?;

    // Runs are keyed by (doc, repo). Fetch per-repo: gather distinct
    // document_repo_ids from the candidate set PLUS all approved repos
    // (so we can show "run not yet attempted" state when a repo exists
    // but no candidates landed).
    let mut repo_ids: std::collections::BTreeSet<Uuid> =
        candidates.iter().map(|c| c.document_repo_id).collect();
    let approved_repos = repo_storage
        .list_for_document(tenant_id, workspace_id, &document_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list document_repos: {e}")))?;
    for r in &approved_repos {
        if matches!(
            r.status,
            edgequake_agents::repo_detection::RepoStatus::Approved
        ) {
            repo_ids.insert(r.id);
        }
    }

    let mut runs = Vec::with_capacity(repo_ids.len());
    for repo_id in repo_ids {
        if let Some(run) = code_storage
            .get_run(tenant_id, workspace_id, &document_id, repo_id)
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to load run: {e}")))?
        {
            runs.push(run.into());
        }
    }

    Ok(Json(CodeReferenceListResponse {
        document_id,
        candidates: candidates.into_iter().map(Into::into).collect(),
        runs,
    }))
}

#[cfg(feature = "postgres")]
async fn review_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    code_artifact_id: Uuid,
    request: ReviewCodeArtifactRequest,
) -> ApiResult<Json<CodeArtifactReviewResponse>> {
    use edgequake_agents::code_analysis::{
        ArtifactStatus, CodeArtifactStorage, PostgresCodeArtifactStorage,
    };

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Code-reference storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresCodeArtifactStorage::new(pool.clone());

    let updated = storage
        .update_status(code_artifact_id, tenant_id, workspace_id, request.status)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to update status: {e}")))?;
    if !updated {
        return Err(ApiError::NotFound(format!(
            "code_artifact {code_artifact_id} not found"
        )));
    }

    // Graph sync: on approve → upsert CODE_FUNCTION node + HAS_REFERENCE_IMPL
    // edge from Algorithm; on reject → drop the edge (keep the row so the
    // user can flip back without losing the artifact). Best-effort: graph
    // failures are logged but don't fail the REST response.
    match request.status {
        ArtifactStatus::Approved | ArtifactStatus::Rejected => {
            let storage_for_graph = PostgresCodeArtifactStorage::new(pool.clone());
            if let Err(e) = sync_graph_for_review(
                &storage_for_graph,
                &state.graph_storage,
                tenant_id,
                workspace_id,
                code_artifact_id,
                request.status,
            )
            .await
            {
                tracing::warn!(
                    error = %e,
                    code_artifact_id = %code_artifact_id,
                    "graph sync failed (non-fatal)"
                );
            }
        }
        ArtifactStatus::Pending => {}
    }

    Ok(Json(CodeArtifactReviewResponse {
        id: code_artifact_id,
        status: types::status_str(request.status),
    }))
}

/// Write or remove the (Algorithm → CODE_FUNCTION) edge in AGE reflecting
/// the review decision. Uses the code_artifact row we just updated to
/// materialise node properties.
#[cfg(feature = "postgres")]
async fn sync_graph_for_review(
    code_storage: &edgequake_agents::code_analysis::PostgresCodeArtifactStorage,
    graph: &std::sync::Arc<dyn edgequake_storage::traits::GraphStorage>,
    tenant_id: Uuid,
    workspace_id: Uuid,
    code_artifact_id: Uuid,
    status: edgequake_agents::code_analysis::ArtifactStatus,
) -> Result<(), String> {
    use edgequake_agents::code_analysis::ArtifactStatus;
    use serde_json::json;
    use std::collections::HashMap;

    // Locate the row so we know the algorithm_id + node props. We query by
    // the natural index on (tenant, workspace, document_id), which we don't
    // have — but list_for_algorithm walks via algorithm_id. Simpler: raw
    // sqlx select by id.
    let url = std::env::var("DATABASE_URL").map_err(|_| "DATABASE_URL not set".to_string())?;
    let pool = sqlx::PgPool::connect(&url)
        .await
        .map_err(|e| format!("connect postgres: {e}"))?;
    let row: Option<(
        Uuid,
        Uuid,
        String,
        String,
        Option<String>,
        String,
        i32,
        i32,
        String,
    )> = sqlx::query_as(
        r#"SELECT algorithm_id, document_repo_id, repo_commit, file_path, symbol_name,
                      language, start_line, end_line, document_id
               FROM code_artifacts
               WHERE id = $1 AND tenant_id = $2 AND workspace_id = $3"#,
    )
    .bind(code_artifact_id)
    .bind(tenant_id)
    .bind(workspace_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| format!("fetch code_artifact: {e}"))?;
    let (
        algorithm_id,
        document_repo_id,
        repo_commit,
        file_path,
        symbol_name,
        language,
        start_line,
        end_line,
        document_id,
    ) = match row {
        Some(r) => r,
        None => return Err(format!("code_artifact {code_artifact_id} disappeared")),
    };
    // Also fetch repo url for useful node props.
    let repo_url: Option<String> =
        sqlx::query_scalar(r#"SELECT url FROM document_repos WHERE id = $1"#)
            .bind(document_repo_id)
            .fetch_optional(&pool)
            .await
            .map_err(|e| format!("fetch document_repos.url: {e}"))?;

    let algo_key = algorithm_id.to_string();
    let code_key = code_artifact_id.to_string();

    match status {
        ArtifactStatus::Approved => {
            let mut props: HashMap<String, serde_json::Value> = HashMap::new();
            props.insert("entity_type".into(), json!("CODE_FUNCTION"));
            props.insert("tenant_id".into(), json!(tenant_id.to_string()));
            props.insert("workspace_id".into(), json!(workspace_id.to_string()));
            props.insert("document_id".into(), json!(document_id));
            props.insert(
                "document_repo_id".into(),
                json!(document_repo_id.to_string()),
            );
            props.insert("algorithm_id".into(), json!(algo_key.clone()));
            props.insert("repo_commit".into(), json!(repo_commit));
            if let Some(url) = repo_url {
                props.insert("repo_url".into(), json!(url));
            }
            props.insert("file_path".into(), json!(file_path));
            if let Some(symbol) = symbol_name {
                props.insert("symbol_name".into(), json!(symbol));
            }
            props.insert("language".into(), json!(language));
            props.insert("start_line".into(), json!(start_line));
            props.insert("end_line".into(), json!(end_line));

            graph
                .upsert_node(&code_key, props)
                .await
                .map_err(|e| format!("upsert CODE_FUNCTION node: {e}"))?;

            let mut edge_props: HashMap<String, serde_json::Value> = HashMap::new();
            edge_props.insert("relation_type".into(), json!("HAS_REFERENCE_IMPL"));
            edge_props.insert("tenant_id".into(), json!(tenant_id.to_string()));
            edge_props.insert("workspace_id".into(), json!(workspace_id.to_string()));
            edge_props.insert("code_artifact_id".into(), json!(code_key.clone()));
            graph
                .upsert_edge(&algo_key, &code_key, edge_props)
                .await
                .map_err(|e| format!("upsert HAS_REFERENCE_IMPL edge: {e}"))?;

            tracing::info!(
                %algorithm_id,
                %code_artifact_id,
                "graph upserted Algorithm → CODE_FUNCTION edge"
            );
        }
        ArtifactStatus::Rejected => {
            graph
                .delete_edge(&algo_key, &code_key)
                .await
                .map_err(|e| format!("delete HAS_REFERENCE_IMPL edge: {e}"))?;
            // Keep the node around; cheap to leave and lets re-approve be
            // a pure edge-upsert without re-building node props.
            tracing::info!(
                %algorithm_id,
                %code_artifact_id,
                "graph dropped HAS_REFERENCE_IMPL edge"
            );
        }
        ArtifactStatus::Pending => {}
    }

    // Silence unused-import warning for code_storage; we might route the
    // lookup through it in a future refactor, but the raw sqlx select above
    // is the simplest path today.
    let _ = code_storage;
    Ok(())
}

// Helper: find a document_repos row by id. Uses list_for_document under the
// hood since PostgresRepoStorage doesn't currently expose a get_by_id.
#[cfg(feature = "postgres")]
async fn find_repo_by_id(
    storage: &edgequake_agents::repo_detection::PostgresRepoStorage,
    tenant_id: Uuid,
    workspace_id: Uuid,
    id: Uuid,
) -> ApiResult<Option<edgequake_agents::repo_detection::DocumentRepo>> {
    use edgequake_agents::repo_detection::RepoStorage;
    // Walk through documents with a matching row — in practice we don't
    // have the document_id yet. Pragmatic shortcut: open a sqlx query that
    // targets the row directly. This reaches into the pool via a tiny raw
    // query to avoid growing the storage trait for a single endpoint.
    let url = std::env::var("DATABASE_URL")
        .map_err(|_| ApiError::Internal("DATABASE_URL not set".to_string()))?;
    let pool = sqlx::PgPool::connect(&url)
        .await
        .map_err(|e| ApiError::Internal(format!("connect postgres: {e}")))?;
    let document_id: Option<String> = sqlx::query_scalar(
        r#"SELECT document_id FROM document_repos
           WHERE id = $1 AND tenant_id = $2 AND workspace_id = $3"#,
    )
    .bind(id)
    .bind(tenant_id)
    .bind(workspace_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| ApiError::Internal(format!("look up document_repo: {e}")))?;
    let Some(doc) = document_id else {
        return Ok(None);
    };
    let rows = storage
        .list_for_document(tenant_id, workspace_id, &doc)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list document_repos: {e}")))?;
    Ok(rows.into_iter().find(|r| r.id == id))
}
