//! Phase 2 reference-codebase RAG endpoints.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

pub fn reference_codebase_routes() -> Router<AppState> {
    Router::new()
        .route("/indexes", post(create_index))
        .route("/indexes/{index_id}", get(get_index))
        .route("/indexes/{index_id}/graph", get(get_index_graph))
        .route("/by-repo/{document_repo_id}", get(list_indexes_by_repo))
        .route("/query", post(query))
}

#[derive(Debug, Deserialize)]
pub struct CreateReferenceCodebaseIndexRequest {
    pub document_repo_id: Uuid,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default)]
    pub force_reindex: bool,
}

#[derive(Debug, Serialize)]
pub struct CreateReferenceCodebaseIndexResponse {
    pub document_repo_id: Uuid,
    pub document_id: String,
    pub track_id: String,
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ReferenceCodebaseIndexResponse {
    pub id: Uuid,
    pub document_id: String,
    pub document_repo_id: Uuid,
    pub repo_url: String,
    pub repo_commit: String,
    pub repo_path: String,
    pub repo_license: Option<String>,
    pub mode: String,
    pub status: String,
    pub language_set: Vec<String>,
    pub file_count: i32,
    pub symbol_count: i32,
    pub chunk_count: i32,
    pub edge_count: i32,
    /// Approximate graph diameter — drives the Code Graph tab's hops
    /// slider max. None for indexes that aren't `complete` yet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<i32>,
    pub error_message: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ReferenceCodebaseQueryRequest {
    pub query: String,
    #[serde(default)]
    pub document_repo_id: Option<Uuid>,
    #[serde(default)]
    pub index_id: Option<Uuid>,
    #[serde(default)]
    pub algorithm_ids: Vec<Uuid>,
    #[serde(default)]
    pub limit: Option<i64>,
    #[serde(default)]
    pub max_distance: Option<f64>,
}

/// Query params for `GET /reference-codebase/indexes/{id}/graph`.
///
/// Exactly one of `anchor_artifact_id` or `anchor_symbol` should be set;
/// if neither is given, the handler falls back to the top-N symbols by
/// degree (useful for exploring a `mode='full'` index without a known
/// entry point).
#[derive(Debug, Deserialize)]
pub struct ReferenceCodebaseGraphParams {
    #[serde(default)]
    pub anchor_artifact_id: Option<Uuid>,
    #[serde(default)]
    pub anchor_symbol: Option<String>,
    #[serde(default)]
    pub hops: Option<u8>,
    #[serde(default)]
    pub max_nodes: Option<u16>,
    /// When true, skip BFS entirely and return the top-N symbols by
    /// degree plus every edge between them. Useful for "show me the
    /// whole repo" exploration. Hard-capped by `max_nodes`.
    #[serde(default)]
    pub whole: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ReferenceCodebaseGraphResponse {
    pub index_id: Uuid,
    pub document_id: String,
    pub repo_url: String,
    pub repo_commit: String,
    pub mode: String,
    pub hops: i32,
    pub truncated: bool,
    pub seed_symbol_ids: Vec<Uuid>,
    pub nodes: Vec<GraphNodeDto>,
    pub edges: Vec<GraphEdgeDto>,
}

#[derive(Debug, Serialize)]
pub struct GraphNodeDto {
    pub symbol_id: Uuid,
    pub name: String,
    pub qualified_name: String,
    pub kind: String,
    pub language: String,
    pub file_path: String,
    pub start_line: i32,
    pub end_line: i32,
    pub depth: i32,
    pub chunk_id: Option<Uuid>,
    pub is_anchor: bool,
    pub algorithm_focus: f32,
    /// Per-language metadata: `{parameters, return_type, docstring, visibility,
    /// is_async, is_test}`. Empty when the extractor had nothing to add.
    #[serde(skip_serializing_if = "serde_json_is_empty_obj")]
    pub metadata: serde_json::Value,
}

fn serde_json_is_empty_obj(v: &serde_json::Value) -> bool {
    matches!(v, serde_json::Value::Object(m) if m.is_empty())
}

#[derive(Debug, Serialize)]
pub struct GraphEdgeDto {
    pub source_symbol_id: Uuid,
    pub target_symbol_id: Uuid,
    pub kind: String,
    pub target_name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ReferenceCodebaseQueryResponse {
    pub coding_context: Vec<CodingContextHit>,
}

#[derive(Debug, Serialize)]
pub struct CodingContextHit {
    pub chunk_id: Uuid,
    pub index_id: Uuid,
    pub document_id: String,
    pub document_repo_id: Uuid,
    pub repo_url: String,
    pub repo_commit: String,
    pub file_path: String,
    pub language: String,
    pub symbol_name: Option<String>,
    pub start_line: i32,
    pub end_line: i32,
    pub chunk_kind: String,
    pub algorithm_id: Option<Uuid>,
    pub content: String,
    pub cosine_distance: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_entity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_score: Option<f64>,
}

fn default_mode() -> String {
    "algorithm_focused".to_string()
}

pub async fn create_index(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Json(request): Json<CreateReferenceCodebaseIndexRequest>,
) -> ApiResult<(StatusCode, Json<CreateReferenceCodebaseIndexResponse>)> {
    #[cfg(feature = "postgres")]
    {
        use edgequake_tasks::{ReferenceCodebaseIndexData, Task, TaskType};

        let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
        let pool = state.pg_pool.as_ref().ok_or_else(|| {
            ApiError::Internal("Reference codebase storage requires PostgreSQL".to_string())
        })?;
        let document_id =
            document_id_for_repo(pool, tenant_id, workspace_id, request.document_repo_id)
                .await?
                .ok_or_else(|| {
                    ApiError::NotFound(format!(
                        "document_repo {} not found",
                        request.document_repo_id
                    ))
                })?;

        let task_data = ReferenceCodebaseIndexData {
            document_id: document_id.clone(),
            workspace_id: workspace_id.to_string(),
            document_repo_id: request.document_repo_id,
            mode: request.mode,
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

        Ok((
            StatusCode::ACCEPTED,
            Json(CreateReferenceCodebaseIndexResponse {
                document_repo_id: request.document_repo_id,
                document_id,
                track_id,
                status: "queued",
            }),
        ))
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, request);
        Err(ApiError::Internal(
            "Reference codebase indexing requires postgres feature".to_string(),
        ))
    }
}

pub async fn get_index(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(index_id): Path<Uuid>,
) -> ApiResult<Json<ReferenceCodebaseIndexResponse>> {
    #[cfg(feature = "postgres")]
    {
        use edgequake_agents::reference_codebase::{
            PostgresReferenceCodebaseStorage, ReferenceCodebaseStorage,
        };
        let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
        let pool = state.pg_pool.as_ref().ok_or_else(|| {
            ApiError::Internal("Reference codebase storage requires PostgreSQL".to_string())
        })?;
        let storage = PostgresReferenceCodebaseStorage::new(pool.clone());
        let mut index = storage
            .get_index(tenant_id, workspace_id, index_id)
            .await
            .map_err(|e| ApiError::Internal(format!("load reference codebase index: {e}")))?
            .ok_or_else(|| {
                ApiError::NotFound(format!("reference codebase index {index_id} not found"))
            })?;
        // Populate approximate graph diameter so the UI can size its
        // hops slider to the codebase. Only makes sense on complete
        // indexes; skip otherwise. Safety-capped at 50 — beyond that
        // we're either in a pathological chain or the query stalled.
        if index.status.as_str() == "complete" && index.edge_count > 0 {
            index.max_depth = storage
                .estimate_max_depth(tenant_id, workspace_id, index_id, 50)
                .await
                .ok()
                .flatten();
        }
        Ok(Json(index.into()))
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, index_id);
        Err(ApiError::Internal(
            "Reference codebase indexes require postgres feature".to_string(),
        ))
    }
}

/// BFS subgraph around an anchor symbol, for the Code Graph tab and the
/// `get_symbol_neighborhood` MCP tool. Seed set resolution, cheapest-first:
///
/// 1. `anchor_artifact_id` → look up overlapping symbols in the index.
/// 2. `anchor_symbol` → exact-or-case-insensitive name match.
/// 3. neither → top-N symbols by degree (fallback for `mode='full'`).
///
/// Rejects when the index is not `status='complete'` — partial builds
/// would render an empty or misleading graph.
pub async fn get_index_graph(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(index_id): Path<Uuid>,
    Query(params): Query<ReferenceCodebaseGraphParams>,
) -> ApiResult<Json<ReferenceCodebaseGraphResponse>> {
    #[cfg(feature = "postgres")]
    {
        use edgequake_agents::reference_codebase::{
            PostgresReferenceCodebaseStorage, ReferenceCodebaseStorage,
        };

        let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
        let pool = state.pg_pool.as_ref().ok_or_else(|| {
            ApiError::Internal("Reference codebase storage requires PostgreSQL".to_string())
        })?;
        let storage = PostgresReferenceCodebaseStorage::new(pool.clone());

        // Default to 3 hops (was 1) so first-click exploration surfaces
        // the typical 2–3 function-call chain around an anchor. No
        // explicit upper clamp — `max_nodes` is the real safety
        // ceiling, and real-world codebases saturate to "basically the
        // whole repo" by hop ~10 anyway. Lower bound stays at 1.
        let hops: i32 = (params.hops.unwrap_or(3) as i32).max(1);
        // Raise the node ceiling for whole-repo exploration (2000
        // keeps V3DB-scale repos fully visible).
        let max_nodes: i32 = params.max_nodes.unwrap_or(200).clamp(1, 2000) as i32;
        let whole = params.whole.unwrap_or(false);

        let index = storage
            .get_index(tenant_id, workspace_id, index_id)
            .await
            .map_err(|e| ApiError::Internal(format!("load index: {e}")))?
            .ok_or_else(|| {
                ApiError::NotFound(format!("reference codebase index {index_id} not found"))
            })?;

        if index.status.as_str() != "complete" {
            return Err(ApiError::BadRequest(format!(
                "index {index_id} is {}, must be 'complete' before graph queries",
                index.status.as_str()
            )));
        }

        // Whole-repo path: skip BFS entirely, seed = top-N symbols by
        // degree (up to max_nodes). Gives the user the full graph in a
        // single request, bounded only by max_nodes. `hops` is ignored.
        if whole {
            let seeds = storage
                .top_symbols_by_degree(tenant_id, workspace_id, index_id, max_nodes as i64)
                .await
                .map_err(|e| ApiError::Internal(format!("top-symbols fallback: {e}")))?;
            if seeds.is_empty() {
                return Ok(Json(ReferenceCodebaseGraphResponse {
                    index_id,
                    document_id: index.document_id,
                    repo_url: index.repo_url,
                    repo_commit: index.repo_commit,
                    mode: index.mode.as_str().to_string(),
                    hops: 0,
                    truncated: false,
                    seed_symbol_ids: Vec::new(),
                    nodes: Vec::new(),
                    edges: Vec::new(),
                }));
            }
            // BFS with hops=0 returns exactly the seed set; we still
            // want that path so we reuse fetch_subgraph's chunk-join.
            let subgraph = storage
                .fetch_subgraph(tenant_id, workspace_id, index_id, &seeds, 0, max_nodes)
                .await
                .map_err(|e| ApiError::Internal(format!("fetch subgraph: {e}")))?;
            // `top_symbols_by_degree` hard-caps at `max_nodes` rows, so
            // `fetch_subgraph` never sees the +1 that would have flagged
            // truncation. Infer from the index's own symbol_count.
            let truncated =
                subgraph.truncated || (subgraph.nodes.len() as i32) < index.symbol_count;
            return Ok(Json(ReferenceCodebaseGraphResponse {
                index_id,
                document_id: index.document_id,
                repo_url: index.repo_url,
                repo_commit: index.repo_commit,
                mode: index.mode.as_str().to_string(),
                hops: 0,
                truncated,
                seed_symbol_ids: seeds,
                nodes: subgraph.nodes.into_iter().map(Into::into).collect(),
                edges: subgraph.edges.into_iter().map(Into::into).collect(),
            }));
        }

        // Seed resolution: artifact id wins if both are given.
        let seeds: Vec<Uuid> = if let Some(artifact_id) = params.anchor_artifact_id {
            let syms = storage
                .anchor_symbols_for_artifact(tenant_id, workspace_id, index_id, artifact_id)
                .await
                .map_err(|e| ApiError::Internal(format!("resolve anchor: {e}")))?;
            if syms.is_empty() {
                return Err(ApiError::NotFound(format!(
                    "code_artifact {artifact_id} did not resolve to any symbol in index {index_id}"
                )));
            }
            syms
        } else if let Some(name) = params
            .anchor_symbol
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let syms = storage
                .symbol_ids_by_name(tenant_id, workspace_id, index_id, name, 10)
                .await
                .map_err(|e| ApiError::Internal(format!("resolve symbol name: {e}")))?;
            if syms.is_empty() {
                return Err(ApiError::NotFound(format!(
                    "symbol '{name}' not found in index {index_id}"
                )));
            }
            syms
        } else {
            // No anchor: fall back to the top-N symbols by degree so the
            // UI has something to render for `mode='full'` exploration.
            storage
                .top_symbols_by_degree(tenant_id, workspace_id, index_id, 20)
                .await
                .map_err(|e| ApiError::Internal(format!("top-symbols fallback: {e}")))?
        };

        if seeds.is_empty() {
            return Ok(Json(ReferenceCodebaseGraphResponse {
                index_id,
                document_id: index.document_id.clone(),
                repo_url: index.repo_url.clone(),
                repo_commit: index.repo_commit.clone(),
                mode: index.mode.as_str().to_string(),
                hops,
                truncated: false,
                seed_symbol_ids: Vec::new(),
                nodes: Vec::new(),
                edges: Vec::new(),
            }));
        }

        let subgraph = storage
            .fetch_subgraph(tenant_id, workspace_id, index_id, &seeds, hops, max_nodes)
            .await
            .map_err(|e| ApiError::Internal(format!("fetch subgraph: {e}")))?;

        Ok(Json(ReferenceCodebaseGraphResponse {
            index_id,
            document_id: index.document_id,
            repo_url: index.repo_url,
            repo_commit: index.repo_commit,
            mode: index.mode.as_str().to_string(),
            hops,
            truncated: subgraph.truncated,
            seed_symbol_ids: seeds,
            nodes: subgraph.nodes.into_iter().map(Into::into).collect(),
            edges: subgraph.edges.into_iter().map(Into::into).collect(),
        }))
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, index_id, params);
        Err(ApiError::Internal(
            "Reference codebase graph requires postgres feature".to_string(),
        ))
    }
}

#[derive(Debug, Serialize)]
pub struct ListIndexesResponse {
    pub indexes: Vec<ReferenceCodebaseIndexResponse>,
}

/// List every reference-codebase index built for a given `document_repo_id`
/// (newest first). Used by the Code Graph tab to discover which index to
/// render when the user has only a repo in hand.
pub async fn list_indexes_by_repo(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_repo_id): Path<Uuid>,
) -> ApiResult<Json<ListIndexesResponse>> {
    #[cfg(feature = "postgres")]
    {
        use edgequake_agents::reference_codebase::{
            PostgresReferenceCodebaseStorage, ReferenceCodebaseStorage,
        };
        let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
        let pool = state.pg_pool.as_ref().ok_or_else(|| {
            ApiError::Internal("Reference codebase storage requires PostgreSQL".to_string())
        })?;
        let storage = PostgresReferenceCodebaseStorage::new(pool.clone());
        let rows = storage
            .list_indexes_for_repo(tenant_id, workspace_id, document_repo_id)
            .await
            .map_err(|e| ApiError::Internal(format!("list indexes: {e}")))?;
        // Enrich complete rows with max_depth so the UI slider sizes
        // to the codebase without an extra round-trip. One CTE per
        // complete index; V3DB-scale repos (6k edges) run in <50ms.
        let mut enriched = Vec::with_capacity(rows.len());
        for mut idx in rows {
            if idx.status.as_str() == "complete" && idx.edge_count > 0 {
                idx.max_depth = storage
                    .estimate_max_depth(tenant_id, workspace_id, idx.id, 50)
                    .await
                    .ok()
                    .flatten();
            }
            enriched.push(idx.into());
        }
        Ok(Json(ListIndexesResponse { indexes: enriched }))
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, document_repo_id);
        Err(ApiError::Internal(
            "Reference codebase storage requires postgres feature".to_string(),
        ))
    }
}

pub async fn query(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Json(request): Json<ReferenceCodebaseQueryRequest>,
) -> ApiResult<Json<ReferenceCodebaseQueryResponse>> {
    #[cfg(feature = "postgres")]
    {
        use edgequake_agents::code_analysis::JinaEmbedder;
        use edgequake_storage::{PgReferenceCodebaseVectorStorage, ReferenceCodebaseVectorStorage};

        let (tenant_id, workspace_id) = parse_tenant_context(&tenant_ctx)?;
        if request.query.trim().is_empty() {
            return Err(ApiError::BadRequest("query is required".to_string()));
        }
        let pool = state.pg_pool.as_ref().ok_or_else(|| {
            ApiError::Internal("Reference codebase storage requires PostgreSQL".to_string())
        })?;
        let code_embed_url = std::env::var("EDGEQUAKE_CODE_EMBEDDING_URL")
            .map_err(|_| ApiError::Internal("EDGEQUAKE_CODE_EMBEDDING_URL not set".to_string()))?;
        let code_model = std::env::var("EDGEQUAKE_CODE_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "jina-code-embeddings".to_string());
        let code_dim: usize = std::env::var("EDGEQUAKE_CODE_EMBEDDING_DIMENSION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(896);
        let embedder = JinaEmbedder::new(code_embed_url, code_model, code_dim);
        let storage = PgReferenceCodebaseVectorStorage::new(pool.clone());
        let limit = request.limit.unwrap_or_else(default_limit);

        // Entity expansion: parse identifiers out of the NL query, fetch their
        // chunks directly. These land ahead of vector hits so exact-name
        // matches never miss just because surrounding prose is sparse.
        let entities = edgequake_agents::reference_codebase::extract_code_entities(&request.query);
        let entity_hits = if entities.is_empty() {
            Vec::new()
        } else {
            storage
                .fetch_chunks_by_symbol_names(
                    tenant_id,
                    workspace_id,
                    &entities,
                    request.document_repo_id,
                    request.index_id,
                    if request.algorithm_ids.is_empty() {
                        None
                    } else {
                        Some(&request.algorithm_ids)
                    },
                    limit,
                )
                .await
                .map_err(|e| ApiError::Internal(format!("entity-expansion lookup failed: {e}")))?
        };

        let query_vec = embedder
            .embed_query_for_code_search(&request.query)
            .await
            .map_err(|e| ApiError::Internal(format!("embed reference codebase query: {e}")))?;
        let vector_hits = storage
            .search_reference_codebase(
                tenant_id,
                workspace_id,
                &query_vec,
                limit,
                request.max_distance.unwrap_or_else(default_max_distance),
                request.document_repo_id,
                request.index_id,
                if request.algorithm_ids.is_empty() {
                    None
                } else {
                    Some(&request.algorithm_ids)
                },
            )
            .await
            .map_err(|e| ApiError::Internal(format!("reference codebase search failed: {e}")))?;

        // Dedup entity + vector candidates by chunk_id (entity wins when both
        // routes surface the same chunk — preserves `matched_entity`).
        let mut seen: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
        let mut merged: Vec<edgequake_storage::ReferenceCodebaseSearchHit> = Vec::new();
        for h in entity_hits.into_iter().chain(vector_hits.into_iter()) {
            if seen.insert(h.chunk_id) {
                merged.push(h);
            }
        }

        // BM25 re-ranking. Build an in-memory corpus from the merged candidates
        // so each chunk competes against the others using a code-aware
        // tokenizer (camelCase / snake_case / digit splits). For identifier
        // queries like "port rebalance_clusters to Rust", BM25 fires on the
        // exact token, complementing cosine similarity.
        //
        // IDF on a candidate-only corpus is weaker than a full-index corpus,
        // but for small K (typically ≤24) it's still directionally right and
        // cheaper than a second SQL roundtrip + full corpus load per query.
        let mut bm25 = edgequake_agents::reference_codebase::Bm25Index::new();
        for h in &merged {
            let blob = format!("{} {}", h.symbol_name.as_deref().unwrap_or(""), h.content);
            bm25.add_document(&h.chunk_id.to_string(), &blob);
        }
        let q_tokens = edgequake_agents::reference_codebase::bm25_tokenize(&request.query);
        let q_refs: Vec<&str> = q_tokens.iter().map(String::as_str).collect();

        for h in merged.iter_mut() {
            let bm = bm25.score_with_tokens_str(&q_refs, &h.chunk_id.to_string());
            // Entity hits come in with cosine_distance = 0.0 (guaranteed best
            // vector component). A matched_entity gives an explicit boost so
            // even if BM25 is low it stays near the top.
            let vector_score = (1.0 - h.cosine_distance.clamp(0.0, 1.0)).max(0.0);
            let entity_boost = if h.matched_entity.is_some() { 0.3 } else { 0.0 };
            let final_score = (0.6 * vector_score + 0.4 * bm + entity_boost).min(1.0);
            h.bm25_score = Some(bm);
            h.final_score = Some(final_score);
        }

        merged.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(limit as usize);

        Ok(Json(ReferenceCodebaseQueryResponse {
            coding_context: merged.into_iter().map(Into::into).collect(),
        }))
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, tenant_ctx, request);
        Err(ApiError::Internal(
            "Reference codebase query requires postgres feature".to_string(),
        ))
    }
}

#[cfg(feature = "postgres")]
fn parse_tenant_context(ctx: &TenantContext) -> ApiResult<(Uuid, Uuid)> {
    let tenant_id = ctx
        .tenant_id
        .as_ref()
        .ok_or_else(|| ApiError::BadRequest("tenant_id is required".to_string()))
        .and_then(|s| {
            Uuid::parse_str(s).map_err(|e| ApiError::BadRequest(format!("Invalid tenant_id: {e}")))
        })?;
    let workspace_id = ctx
        .workspace_id
        .as_ref()
        .ok_or_else(|| ApiError::BadRequest("workspace_id is required".to_string()))
        .and_then(|s| {
            Uuid::parse_str(s)
                .map_err(|e| ApiError::BadRequest(format!("Invalid workspace_id: {e}")))
        })?;
    Ok((tenant_id, workspace_id))
}

#[cfg(feature = "postgres")]
async fn document_id_for_repo(
    pool: &sqlx::PgPool,
    tenant_id: Uuid,
    workspace_id: Uuid,
    document_repo_id: Uuid,
) -> ApiResult<Option<String>> {
    let row = sqlx::query_scalar::<_, String>(
        r#"
        SELECT document_id
        FROM document_repos
        WHERE tenant_id = $1 AND workspace_id = $2 AND id = $3 AND status = 'approved'
        "#,
    )
    .bind(tenant_id)
    .bind(workspace_id)
    .bind(document_repo_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| ApiError::Internal(format!("load document_repo: {e}")))?;
    Ok(row)
}

fn default_limit() -> i64 {
    std::env::var("EDGEQUAKE_REFERENCE_CODEBASE_QUERY_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12)
}

fn default_max_distance() -> f64 {
    std::env::var("EDGEQUAKE_REFERENCE_CODEBASE_QUERY_MAX_DISTANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.65)
}

impl From<edgequake_agents::reference_codebase::CodebaseIndex> for ReferenceCodebaseIndexResponse {
    fn from(i: edgequake_agents::reference_codebase::CodebaseIndex) -> Self {
        Self {
            id: i.id,
            document_id: i.document_id,
            document_repo_id: i.document_repo_id,
            repo_url: i.repo_url,
            repo_commit: i.repo_commit,
            repo_path: i.repo_path,
            repo_license: i.repo_license,
            mode: i.mode.as_str().to_string(),
            status: i.status.as_str().to_string(),
            language_set: i.language_set,
            file_count: i.file_count,
            symbol_count: i.symbol_count,
            chunk_count: i.chunk_count,
            edge_count: i.edge_count,
            max_depth: i.max_depth,
            error_message: i.error_message,
        }
    }
}

impl From<edgequake_agents::reference_codebase::SubgraphNode> for GraphNodeDto {
    fn from(n: edgequake_agents::reference_codebase::SubgraphNode) -> Self {
        Self {
            symbol_id: n.symbol_id,
            name: n.name,
            qualified_name: n.qualified_name,
            kind: n.kind,
            language: n.language,
            file_path: n.file_path,
            start_line: n.start_line,
            end_line: n.end_line,
            depth: n.depth,
            chunk_id: n.chunk_id,
            is_anchor: n.is_anchor,
            algorithm_focus: n.algorithm_focus,
            metadata: n.metadata,
        }
    }
}

impl From<edgequake_agents::reference_codebase::SubgraphEdge> for GraphEdgeDto {
    fn from(e: edgequake_agents::reference_codebase::SubgraphEdge) -> Self {
        Self {
            source_symbol_id: e.source_symbol_id,
            target_symbol_id: e.target_symbol_id,
            kind: e.kind,
            target_name: e.target_name,
        }
    }
}

impl From<edgequake_storage::ReferenceCodebaseSearchHit> for CodingContextHit {
    fn from(h: edgequake_storage::ReferenceCodebaseSearchHit) -> Self {
        Self {
            chunk_id: h.chunk_id,
            index_id: h.index_id,
            document_id: h.document_id,
            document_repo_id: h.document_repo_id,
            repo_url: h.repo_url,
            repo_commit: h.repo_commit,
            file_path: h.file_path,
            language: h.language,
            symbol_name: h.symbol_name,
            start_line: h.start_line,
            end_line: h.end_line,
            chunk_kind: h.chunk_kind,
            algorithm_id: h.algorithm_id,
            content: h.content,
            cosine_distance: h.cosine_distance,
            matched_entity: h.matched_entity,
            bm25_score: h.bm25_score,
            final_score: h.final_score,
        }
    }
}
