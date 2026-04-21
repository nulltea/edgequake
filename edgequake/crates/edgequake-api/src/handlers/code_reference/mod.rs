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
        .route("/by-document/{document_id}/submit", post(submit))
        .route("/counts", get(counts))
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

/// Workspace-scoped code-artifact counts, grouped by document.
///
/// Used by the document-list UI to decide whether to render the per-row
/// "Reference code" action button — no row for a doc here → no button.
pub async fn counts(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
) -> ApiResult<Json<CodeArtifactCountsResponse>> {
    #[cfg(feature = "postgres")]
    {
        use edgequake_agents::code_analysis::{CodeArtifactStorage, PostgresCodeArtifactStorage};

        let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
        let pool = state.pg_pool.as_ref().ok_or_else(|| {
            ApiError::Internal("Code-reference counts require PostgreSQL pool".to_string())
        })?;
        let storage = PostgresCodeArtifactStorage::new(pool.clone());
        let rows = storage
            .counts_by_document(tenant_id, workspace_id)
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to count code_artifacts: {e}")))?;
        let counts = rows
            .into_iter()
            .map(|(document_id, count)| CodeArtifactCountEntry { document_id, count })
            .collect();
        Ok(Json(CodeArtifactCountsResponse { counts }))
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx);
        Err(ApiError::Internal(
            "Code-reference counts require the postgres feature".to_string(),
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

/// Finalise the review — rejects that every artifact is still Pending,
/// then auto-enqueues a `ReferenceCodebaseIndex` task for each distinct
/// `(document_repo, commit)` with approved rows (gated by
/// `auto_index_enabled()`). Mirrors `POST /algorithms/by-document/{id}/submit`.
pub async fn submit(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
) -> ApiResult<Json<CodeReferenceSubmitResponse>> {
    #[cfg(feature = "postgres")]
    {
        submit_impl(state, tenant_ctx, document_id).await
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, document_id);
        Err(ApiError::Internal(
            "Code-reference submit requires the postgres feature".to_string(),
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
            // Embedding sync: approve → embed + index; reject → delete index.
            // Best-effort, same pattern: failures don't poison the review.
            if let Err(e) = sync_embedding_for_review(
                pool,
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
                    "embedding sync failed (non-fatal)"
                );
            }

            // Phase 2 auto-enqueue now runs from the `submit` handler
            // (POST /by-document/{id}/submit), not per-approve — mirrors
            // the algorithms review → submit → embed flow. Keeps the
            // review path cheap and lets the user finish triaging all
            // matches before kicking off indexing.
        }
        ArtifactStatus::Pending => {}
    }

    Ok(Json(CodeArtifactReviewResponse {
        id: code_artifact_id,
        status: types::status_str(request.status),
    }))
}

#[cfg(feature = "postgres")]
async fn submit_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    document_id: String,
) -> ApiResult<Json<CodeReferenceSubmitResponse>> {
    use edgequake_agents::code_analysis::{
        ArtifactStatus, CodeArtifactStorage, PostgresCodeArtifactStorage,
    };

    let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Code-reference storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PostgresCodeArtifactStorage::new(pool.clone());

    let candidates = storage
        .list_for_document(tenant_id, workspace_id, &document_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list code_artifacts: {e}")))?;

    if candidates.is_empty() {
        return Err(ApiError::BadRequest(
            "no code matches to submit for this document".to_string(),
        ));
    }

    let pending_count = candidates
        .iter()
        .filter(|c| c.status == ArtifactStatus::Pending)
        .count();
    if pending_count > 0 {
        return Err(ApiError::BadRequest(format!(
            "{pending_count} code match(es) still pending review — review every match before submitting"
        )));
    }

    // Pick one approved artifact per distinct document_repo. Since
    // `maybe_auto_enqueue_index` is keyed on (document_repo, commit,
    // mode) and `INSERT … ON CONFLICT DO NOTHING`, any approved row
    // from the same repo would trigger the same insert — grouping by
    // repo just avoids redundant work.
    let mut seen_repos: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    let mut anchors: Vec<Uuid> = Vec::new();
    let mut approved_count: usize = 0;
    for c in candidates.iter() {
        if c.status != ArtifactStatus::Approved {
            continue;
        }
        approved_count += 1;
        if seen_repos.insert(c.document_repo_id) {
            anchors.push(c.id);
        }
    }
    let rejected_count = candidates.len() - approved_count;

    if approved_count == 0 {
        return Ok(Json(CodeReferenceSubmitResponse {
            document_id,
            approved_count: 0,
            rejected_count,
            indexes_queued: 0,
            status: "no_approved_matches",
        }));
    }

    if !auto_index_enabled() {
        info!(
            document_id = %document_id,
            approved_count,
            rejected_count,
            "Code-reference submit: auto-index gate is off; approved without queueing indexes"
        );
        return Ok(Json(CodeReferenceSubmitResponse {
            document_id,
            approved_count,
            rejected_count,
            indexes_queued: 0,
            status: "auto_index_disabled",
        }));
    }

    let mut indexes_queued: usize = 0;
    for anchor in anchors {
        match maybe_auto_enqueue_index(
            pool,
            &state.task_storage,
            &state.task_queue,
            tenant_id,
            workspace_id,
            anchor,
        )
        .await
        {
            // Only count true enqueues — the helper logs on its own when
            // it skips (existing index) but returns Ok(()) either way,
            // so we can't distinguish here. For a user-facing number
            // we'd need it to return an enum; for now we optimistically
            // count each successful helper call as "queued or already".
            Ok(()) => indexes_queued += 1,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    code_artifact_id = %anchor,
                    "reference-codebase auto-enqueue from submit failed (non-fatal)"
                );
            }
        }
    }

    info!(
        document_id = %document_id,
        approved_count,
        rejected_count,
        indexes_queued,
        "Code-reference submit: indexing kicked off"
    );

    Ok(Json(CodeReferenceSubmitResponse {
        document_id,
        approved_count,
        rejected_count,
        indexes_queued,
        status: "indexing_queued",
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

/// Embed the approved snippet into `code_artifact_embeddings` (upsert) or
/// drop its index row on rejection. Silently skipped when
/// `EDGEQUAKE_CODE_EMBEDDING_URL` is unset — lets the feature roll out
/// without a hard dep on the embedder service.
#[cfg(feature = "postgres")]
async fn sync_embedding_for_review(
    pool: &sqlx::PgPool,
    tenant_id: Uuid,
    workspace_id: Uuid,
    code_artifact_id: Uuid,
    status: edgequake_agents::code_analysis::ArtifactStatus,
) -> Result<(), String> {
    use edgequake_agents::code_analysis::{ArtifactStatus, CodeEmbeddingStorage, JinaEmbedder};

    let storage = CodeEmbeddingStorage::new(pool.clone());

    match status {
        ArtifactStatus::Approved => {
            let Some(url) = std::env::var("EDGEQUAKE_CODE_EMBEDDING_URL")
                .ok()
                .filter(|v| !v.is_empty())
            else {
                tracing::debug!(
                    code_artifact_id = %code_artifact_id,
                    "EDGEQUAKE_CODE_EMBEDDING_URL unset — skipping embedding"
                );
                return Ok(());
            };
            let model = std::env::var("EDGEQUAKE_CODE_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "jina-code-embeddings".to_string());
            let dim: usize = std::env::var("EDGEQUAKE_CODE_EMBEDDING_DIMENSION")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(896);

            // Load the snippet + metadata we need to vectorise + upsert.
            let row: Option<(String, Uuid, String)> = sqlx::query_as(
                r#"SELECT snippet, algorithm_id, document_id
                   FROM code_artifacts
                   WHERE id = $1 AND tenant_id = $2 AND workspace_id = $3"#,
            )
            .bind(code_artifact_id)
            .bind(tenant_id)
            .bind(workspace_id)
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("fetch code_artifact: {e}"))?;
            let (snippet, algorithm_id, document_id) =
                row.ok_or_else(|| format!("code_artifact {code_artifact_id} disappeared"))?;

            let embedder = JinaEmbedder::new(url, &model, dim);
            let embedding = embedder
                .embed_code_for_indexing(&snippet)
                .await
                .map_err(|e| format!("embed snippet: {e}"))?;

            let embedding_row_id = storage
                .upsert(
                    code_artifact_id,
                    tenant_id,
                    workspace_id,
                    &document_id,
                    algorithm_id,
                    &model,
                    &embedding,
                )
                .await
                .map_err(|e| format!("persist embedding: {e}"))?;

            tracing::info!(
                %code_artifact_id,
                %embedding_row_id,
                dim = embedding.len(),
                "indexed approved code_artifact into vector store"
            );
        }
        ArtifactStatus::Rejected => {
            if let Err(e) = storage.delete_for_artifact(code_artifact_id).await {
                return Err(format!("delete embedding: {e}"));
            }
            tracing::info!(
                %code_artifact_id,
                "removed rejected code_artifact from vector store"
            );
        }
        ArtifactStatus::Pending => {}
    }
    Ok(())
}

/// Read `EDGEQUAKE_REFERENCE_CODEBASE_AUTO_INDEX` (default `false`). Gate
/// for the Phase 2 auto-enqueue side-effect. Kept a free function so a
/// flag flip can happen via env without a code edit.
#[cfg(feature = "postgres")]
fn auto_index_enabled() -> bool {
    matches!(
        std::env::var("EDGEQUAKE_REFERENCE_CODEBASE_AUTO_INDEX")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Race-safe auto-enqueue of a `reference_codebase_index` task after a
/// code_artifact was approved. Resolves `(document_repo_id, repo_commit)`
/// from the artifact + its repo row, then `INSERT … ON CONFLICT DO NOTHING
/// RETURNING id` against the `(tenant, workspace, document_repo, commit,
/// mode)` unique key — if an index already exists in any state
/// (queued/in-flight/complete/failed), the INSERT returns no row and we
/// skip enqueuing.
///
/// Only fires when the row is genuinely new, so concurrent approvals
/// across sibling artifacts pointing at the same repo enqueue exactly
/// one indexing task.
///
/// Snapshots `repo_commit` at enqueue time and writes it into the index
/// row — never relies on re-reading it from `document_repos` later, so
/// the detection pipeline drifting HEAD can't silently bypass the unique
/// constraint.
#[cfg(feature = "postgres")]
async fn maybe_auto_enqueue_index(
    pool: &sqlx::PgPool,
    task_storage: &edgequake_tasks::SharedTaskStorage,
    task_queue: &edgequake_tasks::SharedTaskQueue,
    tenant_id: Uuid,
    workspace_id: Uuid,
    code_artifact_id: Uuid,
) -> Result<(), String> {
    use edgequake_tasks::{ReferenceCodebaseIndexData, Task, TaskType};

    // Look up the artifact to resolve (document_id, document_repo_id,
    // repo_commit). Approved status is implied by the caller — we only
    // hit this path from the approval handler after a successful status
    // transition.
    let row: Option<(String, Uuid, String)> = sqlx::query_as(
        r#"
        SELECT document_id, document_repo_id, repo_commit
        FROM code_artifacts
        WHERE id = $1 AND tenant_id = $2 AND workspace_id = $3
        "#,
    )
    .bind(code_artifact_id)
    .bind(tenant_id)
    .bind(workspace_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("lookup code_artifact: {e}"))?;

    let Some((document_id, document_repo_id, repo_commit)) = row else {
        // Artifact was deleted out from under us (e.g. algorithm
        // re-extraction); nothing to enqueue.
        return Ok(());
    };

    // Resolve repo_url for the index row (snapshot — same reason we
    // snapshot commit: document_repos mutates independently).
    let repo_url: Option<String> = sqlx::query_scalar(
        r#"
        SELECT url FROM document_repos
        WHERE id = $1 AND tenant_id = $2 AND workspace_id = $3 AND status = 'approved'
        "#,
    )
    .bind(document_repo_id)
    .bind(tenant_id)
    .bind(workspace_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("lookup document_repo: {e}"))?;

    let Some(repo_url) = repo_url else {
        // Repo not approved (or gone). Normal path for approvals on
        // orphaned artifacts — skip silently.
        return Ok(());
    };

    // Auto-triggered indexes get a conservative file cap to protect
    // against "click-approve on a 20k-file repo burns half the Jina
    // quota". Explicit POST requests can raise it via force_reindex.
    let max_files_override: i32 = std::env::var("EDGEQUAKE_REFERENCE_CODEBASE_AUTO_MAX_FILES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5000);

    // Race-safe insert. `repo_path` is '' here — the indexer overwrites
    // it with the real `/workspace/<sha>` path once the code-analyzer
    // /snapshot call returns.
    let inserted: Option<Uuid> = sqlx::query_scalar(
        r#"
        INSERT INTO reference_codebase_indexes (
            tenant_id, workspace_id, document_id, document_repo_id,
            repo_url, repo_commit, repo_path, repo_license,
            mode, status, auto_triggered, max_files_override
        )
        VALUES (
            $1, $2, $3, $4,
            $5, $6, '', NULL,
            'algorithm_focused', 'queued', TRUE, $7
        )
        ON CONFLICT (tenant_id, workspace_id, document_repo_id, repo_commit, mode)
        DO NOTHING
        RETURNING id
        "#,
    )
    .bind(tenant_id)
    .bind(workspace_id)
    .bind(&document_id)
    .bind(document_repo_id)
    .bind(&repo_url)
    .bind(&repo_commit)
    .bind(max_files_override)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("insert reference_codebase_index: {e}"))?;

    let Some(_index_id) = inserted else {
        tracing::debug!(
            %code_artifact_id,
            %document_repo_id,
            %repo_commit,
            "reference-codebase index already exists for this (repo, commit) — auto-enqueue skipped"
        );
        return Ok(());
    };

    // The processor resolves the index row by (document_repo_id, commit,
    // mode); we don't need to plumb the new id through.
    let task_data = ReferenceCodebaseIndexData {
        document_id: document_id.clone(),
        workspace_id: workspace_id.to_string(),
        document_repo_id,
        mode: "algorithm_focused".to_string(),
        force_reindex: false,
    };
    let task = Task::new(
        tenant_id,
        workspace_id,
        TaskType::ReferenceCodebaseIndex,
        serde_json::to_value(&task_data).map_err(|e| format!("serialize task: {e}"))?,
    );
    let track_id = task.track_id.clone();
    task_storage
        .create_task(&task)
        .await
        .map_err(|e| format!("create task: {e}"))?;
    task_queue
        .send(task)
        .await
        .map_err(|e| format!("queue task: {e}"))?;

    tracing::info!(
        %code_artifact_id,
        %document_repo_id,
        %repo_commit,
        %track_id,
        "auto-enqueued reference-codebase index task after code_artifact approval"
    );
    Ok(())
}
