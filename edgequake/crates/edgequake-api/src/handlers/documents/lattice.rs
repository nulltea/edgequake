//! Plan-6 lattice-facing handlers:
//!   POST   /api/v1/documents/raw                  — raw kv_storage write
//!   POST   /api/v1/documents/{doc_id}/chunks      — pre-chunked vectors
//!   PATCH  /api/v1/documents/{doc_id}             — status update
//!
//! These exist so lattice can populate edgequake's `/documents` page and
//! `workspace_vector_storage` without invoking edgequake's full ingestion
//! pipeline (which would re-extract entities lattice has already extracted
//! itself via Claude Agent SDK).

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::handlers::documents::storage_helpers::get_workspace_vector_storage_strict;
use crate::handlers::documents_types::{
    CreateChunksRequest, CreateChunksResponse, RawDocumentRequest, RawDocumentResponse,
    UpdateDocumentRequest, UpdateDocumentResponse,
};
use crate::handlers::entities::embed::embed_text_via_workspace;
use crate::handlers::workspaces::invalidate_workspace_stats_cache;
use crate::middleware::TenantContext;
use crate::services::ContentHasher;
use crate::state::AppState;

/// `POST /api/v1/documents/raw` — create a kv_storage entry without
/// triggering chunking/extraction. Body lives at `{doc_id}-content`,
/// metadata at `{doc_id}-metadata`. Returns the doc UUID.
#[utoipa::path(
    post,
    path = "/api/v1/documents/raw",
    tag = "Documents",
    request_body = RawDocumentRequest,
    responses(
        (status = 201, description = "Raw document stored", body = RawDocumentResponse),
        (status = 400, description = "Invalid request"),
    ),
)]
pub async fn create_raw_document(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Json(req): Json<RawDocumentRequest>,
) -> ApiResult<(StatusCode, Json<RawDocumentResponse>)> {
    debug!(
        tenant_id = ?tenant_ctx.tenant_id,
        workspace_id = ?tenant_ctx.workspace_id,
        content_len = req.content.len(),
        "POST /documents/raw — lattice fast-path"
    );

    crate::validation::validate_content(&req.content, state.config.max_document_size)?;

    let document_id = Uuid::new_v4().to_string();
    let content_hash = ContentHasher::hash_str(&req.content);
    let workspace_id = tenant_ctx
        .workspace_id
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let tenant_id = tenant_ctx.tenant_id.clone();

    let status = req
        .status
        .clone()
        .unwrap_or_else(|| "processing".to_string());
    let now = Utc::now().to_rfc3339();
    let content_length = req.content.len();
    let content_summary = crate::validation::generate_content_summary(&req.content);

    // Mirror text_upload.rs's metadata shape so /documents-page consumers
    // see lattice-written rows the same as pipeline-written ones.
    let doc_metadata = serde_json::json!({
        "id": document_id,
        "title": req.title,
        "content_summary": content_summary,
        "content_length": content_length,
        "content_hash": content_hash,
        "file_size_bytes": content_length,
        "sha256_checksum": content_hash,
        "document_type": "markdown",
        "track_id": req.source_id,
        "created_at": now,
        "updated_at": now,
        "status": status,
        "tenant_id": tenant_id,
        "workspace_id": workspace_id,
        "source_type": "markdown",
        // current_stage must be one of the frontend's known states
        // (pending|processing|chunking|extracting|embedding|indexing|
        // completed|indexed|failed) — the documents-page status badge
        // reads current_stage in preference to `status`, and falls
        // through to "pending" for any value it doesn't recognise.
        // Mirror the initial `status` here so the badge agrees.
        "current_stage": status,
        "stage_progress": 0.0,
        "stage_message": "Document body stored; extraction handled by lattice client",
    });

    let metadata_key = format!("{}-metadata", document_id);
    let content_key = format!("{}-content", document_id);

    state
        .kv_storage
        .upsert(&[
            (metadata_key, doc_metadata),
            (
                content_key,
                serde_json::json!({"content": req.content}),
            ),
        ])
        .await?;

    // Also insert a row into the relational `documents` table so the
    // `chunks.document_id` FK can be satisfied when create_chunks
    // mirrors table-kind chunks. Pipeline writes do this implicitly via
    // text_upload.rs; the lattice fast-path bypasses that path, so we
    // mirror the minimum schema here. Best-effort: a failure here
    // doesn't fail the kv write (older deployments without the table
    // would otherwise lose all lattice docs).
    #[cfg(feature = "postgres")]
    if let Err(e) =
        upsert_document_row(&state, &document_id, &req, &workspace_id, &tenant_id, &status, &content_hash)
            .await
    {
        warn!(
            document_id = %document_id,
            error = %e,
            "Failed to upsert documents row (chunks FK mirror will skip)"
        );
    }

    if let Ok(ws_uuid) = Uuid::parse_str(&workspace_id) {
        invalidate_workspace_stats_cache(ws_uuid).await;
    }

    Ok((
        StatusCode::CREATED,
        Json(RawDocumentResponse {
            document_id,
            status,
        }),
    ))
}

/// `POST /api/v1/documents/{doc_id}/chunks` — accept lattice-prepared
/// chunks, optionally embed each via the workspace provider, store the
/// resulting `(id, vector, metadata{type:"chunk", ...})` in
/// `workspace_vector_storage`.
#[utoipa::path(
    post,
    path = "/api/v1/documents/{doc_id}/chunks",
    tag = "Documents",
    request_body = CreateChunksRequest,
    responses(
        (status = 200, description = "Chunks stored", body = CreateChunksResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Document not found"),
    ),
)]
pub async fn create_chunks(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(doc_id): Path<String>,
    Json(req): Json<CreateChunksRequest>,
) -> ApiResult<Json<CreateChunksResponse>> {
    let metadata_key = format!("{}-metadata", doc_id);
    let metadata = state
        .kv_storage
        .get_by_ids(std::slice::from_ref(&metadata_key))
        .await?
        .into_iter()
        .next()
        .filter(|v| !v.is_null())
        .ok_or_else(|| ApiError::NotFound(format!("Document {} not found", doc_id)))?;

    enforce_tenant_match(&metadata, &tenant_ctx)?;

    let workspace_id = tenant_ctx
        .workspace_id
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let tenant_id = tenant_ctx.tenant_id.clone();

    let mut chunks_stored = 0usize;
    let mut embeddings_generated = 0usize;

    if !req.chunks.is_empty() {
        // Server-side embedding requires the workspace's embedding provider;
        // resolved once per request.
        let vector_storage = get_workspace_vector_storage_strict(&state, &workspace_id).await?;

        let mut upsert_batch: Vec<(String, Vec<f32>, serde_json::Value)> =
            Vec::with_capacity(req.chunks.len());
        // Also mirror the pipeline's kv layout: one entry per chunk under
        // `{doc_id}-chunk-{N}` so the /documents page's chunk-count scan
        // and the recovery handler both see lattice-uploaded chunks the
        // same as pipeline-uploaded ones.
        let mut kv_chunk_batch: Vec<(String, serde_json::Value)> =
            Vec::with_capacity(req.chunks.len());

        for chunk in &req.chunks {
            // Merge lattice's per-chunk metadata with the server-set fields.
            let mut metadata = match &chunk.metadata {
                serde_json::Value::Object(m) => m.clone(),
                _ => serde_json::Map::new(),
            };
            metadata.insert("type".to_string(), serde_json::json!("chunk"));
            metadata.insert("document_id".to_string(), serde_json::json!(doc_id));
            metadata.insert("workspace_id".to_string(), serde_json::json!(&workspace_id));
            if let Some(ref tid) = tenant_id {
                metadata.insert("tenant_id".to_string(), serde_json::json!(tid));
            }
            metadata.insert("content".to_string(), serde_json::json!(&chunk.content));
            let metadata_value = serde_json::Value::Object(metadata.clone());

            // Pull chunk_index from lattice's metadata so the kv key matches
            // edgequake's pipeline-native convention.
            let chunk_index = metadata
                .get("chunk_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(chunks_stored as u64);
            let kv_key = format!("{}-chunk-{}", doc_id, chunk_index);
            let mut kv_value_map = metadata.clone();
            kv_value_map.insert("vector_id".to_string(), serde_json::json!(&chunk.id));
            kv_chunk_batch.push((kv_key, serde_json::Value::Object(kv_value_map)));

            if chunk.embed.unwrap_or(false) {
                let text = chunk
                    .embedding_text
                    .clone()
                    .unwrap_or_else(|| chunk.content.clone());
                match embed_text_via_workspace(&state, &workspace_id, &text).await {
                    Ok(Some(vector)) => {
                        upsert_batch.push((chunk.id.clone(), vector, metadata_value));
                        embeddings_generated += 1;
                    }
                    Ok(None) => {
                        // Workspace has no embedding provider configured;
                        // skip silently but log so the user can fix.
                        warn!(
                            chunk_id = %chunk.id,
                            workspace_id = %workspace_id,
                            "Skipping chunk embed: workspace has no embedding provider configured"
                        );
                    }
                    Err(e) => {
                        warn!(chunk_id = %chunk.id, error = %e, "Failed to embed chunk");
                    }
                }
            }
            chunks_stored += 1;
        }

        if !kv_chunk_batch.is_empty() {
            state.kv_storage.upsert(&kv_chunk_batch).await.map_err(|e| {
                ApiError::Internal(format!("Failed to upsert chunk kv records: {}", e))
            })?;
        }
        if !upsert_batch.is_empty() {
            vector_storage.upsert(&upsert_batch).await.map_err(|e| {
                ApiError::Internal(format!("Failed to upsert chunk vectors: {}", e))
            })?;
        }

        // Surface table chunks in the Figures tab by mirroring them into
        // the `chunks` SQL table with `kind = 'table'`. Markdown tables
        // come through with `metadata.element_kind == "table"`; we parse
        // the markdown table on the way in so the table-detail page can
        // render structured rows + HTML the same way it does for VLM-OCR
        // tables from the PDF pipeline.
        #[cfg(feature = "postgres")]
        if let Err(e) = persist_table_chunks(&state, &doc_id, &tenant_id, &workspace_id, &req.chunks).await {
            warn!(document_id = %doc_id, error = %e, "Failed to mirror table chunks to chunks table");
        }
    }

    if let Ok(ws_uuid) = Uuid::parse_str(&workspace_id) {
        invalidate_workspace_stats_cache(ws_uuid).await;
    }

    Ok(Json(CreateChunksResponse {
        document_id: doc_id,
        chunks_stored,
        embeddings_generated,
    }))
}

/// `PATCH /api/v1/documents/{doc_id}` — update mutable fields on the
/// kv_storage metadata entry. Lattice uses this to move docs from
/// "processing" → "completed"/"failed" after its own extraction finishes.
#[utoipa::path(
    patch,
    path = "/api/v1/documents/{doc_id}",
    tag = "Documents",
    request_body = UpdateDocumentRequest,
    responses(
        (status = 200, description = "Document updated", body = UpdateDocumentResponse),
        (status = 404, description = "Document not found"),
    ),
)]
pub async fn patch_document(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(doc_id): Path<String>,
    Json(req): Json<UpdateDocumentRequest>,
) -> ApiResult<Json<UpdateDocumentResponse>> {
    let metadata_key = format!("{}-metadata", doc_id);
    let mut metadata = state
        .kv_storage
        .get_by_ids(std::slice::from_ref(&metadata_key))
        .await?
        .into_iter()
        .next()
        .filter(|v| !v.is_null())
        .ok_or_else(|| ApiError::NotFound(format!("Document {} not found", doc_id)))?;

    enforce_tenant_match(&metadata, &tenant_ctx)?;

    let (resulting_status, workspace_id) = {
        let obj = metadata
            .as_object_mut()
            .ok_or_else(|| ApiError::Internal("Doc metadata is not an object".into()))?;

        if let Some(ref status) = req.status {
            // Auto-set stage_progress to 1.0 on terminal-success states so
            // the UI's progress-based "pending" / "in-progress" gating
            // doesn't keep showing the doc as unfinished after lattice
            // marks it completed.
            if status == "completed" {
                obj.insert("stage_progress".to_string(), serde_json::json!(1.0));
            }
            obj.insert("status".to_string(), serde_json::json!(status));
            // Mirror status → current_stage when the caller didn't send an
            // explicit current_stage. The documents page's status badge
            // reads current_stage in preference to status; if we leave
            // current_stage at its initial value, the badge never flips
            // to "completed" even after extraction succeeds.
            if req.current_stage.is_none() {
                obj.insert("current_stage".to_string(), serde_json::json!(status));
            }
        }
        if let Some(ref stage) = req.current_stage {
            obj.insert("current_stage".to_string(), serde_json::json!(stage));
        }
        if let Some(ref msg) = req.stage_message {
            obj.insert("stage_message".to_string(), serde_json::json!(msg));
        }
        if let Some(n) = req.entity_count {
            obj.insert("entity_count".to_string(), serde_json::json!(n));
        }
        if let Some(n) = req.relationship_count {
            obj.insert("relationship_count".to_string(), serde_json::json!(n));
        }
        // Extended cost / model / timing metadata that the /documents page
        // surfaces as columns (and that the detail page renders alongside
        // the doc body). All fields are optional and only overwrite when
        // sent.
        if let Some(n) = req.input_tokens {
            obj.insert("input_tokens".to_string(), serde_json::json!(n));
        }
        if let Some(n) = req.output_tokens {
            obj.insert("output_tokens".to_string(), serde_json::json!(n));
        }
        if let Some(n) = req.total_tokens {
            obj.insert("total_tokens".to_string(), serde_json::json!(n));
        }
        if let Some(v) = req.cost_usd {
            obj.insert("cost_usd".to_string(), serde_json::json!(v));
        }
        if let Some(ref s) = req.llm_model {
            obj.insert("llm_model".to_string(), serde_json::json!(s));
        }
        if let Some(ref s) = req.embedding_model {
            obj.insert("embedding_model".to_string(), serde_json::json!(s));
        }
        if let Some(ms) = req.processing_duration_ms {
            obj.insert(
                "processing_duration_ms".to_string(),
                serde_json::json!(ms),
            );
        }
        obj.insert(
            "updated_at".to_string(),
            serde_json::json!(Utc::now().to_rfc3339()),
        );

        let status = obj
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let ws_id = obj
            .get("workspace_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        (status, ws_id)
    };

    state
        .kv_storage
        .upsert(&[(metadata_key, metadata)])
        .await?;

    // Mirror status (+ counts + cost telemetry) into the SQL `documents`
    // row so the frontend, which reads `documents.status` for the
    // /documents page, sees the same state as kv_storage. Without this,
    // lattice docs sit on "processing" forever in the UI even though
    // kv-side patches landed and downstream extraction finished.
    //
    // Best-effort: a failure to update the SQL row doesn't fail the
    // PATCH (kv stays canonical). Logged at warn so it's visible.
    #[cfg(feature = "postgres")]
    if let Err(e) = update_documents_row_from_patch(&state, &doc_id, &req).await {
        warn!(
            document_id = %doc_id,
            error = %e,
            "Failed to mirror PATCH fields to documents SQL row"
        );
    }

    if let Some(ws_id) = workspace_id {
        if let Ok(ws_uuid) = Uuid::parse_str(&ws_id) {
            invalidate_workspace_stats_cache(ws_uuid).await;
        }
    }

    Ok(Json(UpdateDocumentResponse {
        document_id: doc_id,
        status: resulting_status,
    }))
}

/// Mirror the patch fields onto the relational `documents` row.
///
/// The frontend's /documents page reads `status`, `entity_count`, and
/// `relationship_count` from `documents`, not from kv_storage. When the
/// lattice fast-path PATCHes status to "completed", kv is updated but the
/// SQL row remains "processing" — UI then shows the doc as pending forever.
///
/// We update only the columns the PATCH actually carries (status, counts).
/// Cost/token/model fields don't have corresponding SQL columns yet —
/// they stay in kv only.
#[cfg(feature = "postgres")]
async fn update_documents_row_from_patch(
    state: &AppState,
    document_id: &str,
    req: &UpdateDocumentRequest,
) -> Result<(), String> {
    // Nothing to mirror? Skip the round-trip.
    if req.status.is_none()
        && req.entity_count.is_none()
        && req.relationship_count.is_none()
    {
        return Ok(());
    }

    let Some(pool) = state.pg_pool.as_ref() else {
        return Err("Postgres pool not configured".to_string());
    };
    let doc_uuid = Uuid::parse_str(document_id).map_err(|e| format!("invalid doc_id: {e}"))?;

    let res = sqlx::query(
        r#"
        UPDATE documents
        SET
            status             = COALESCE($2, status),
            entity_count       = COALESCE($3, entity_count),
            relationship_count = COALESCE($4, relationship_count),
            updated_at         = now()
        WHERE id = $1
        "#,
    )
    .bind(doc_uuid)
    .bind(req.status.as_deref())
    .bind(req.entity_count.map(|n| n as i32))
    .bind(req.relationship_count.map(|n| n as i32))
    .execute(pool)
    .await
    .map_err(|e| format!("UPDATE documents failed: {e}"))?;

    if res.rows_affected() == 0 {
        // Row not present (e.g. an older lattice doc created before
        // upsert_document_row landed). Not an error — kv is still
        // canonical.
        return Ok(());
    }
    Ok(())
}

/// Insert/upsert a row into the relational `documents` table so the
/// `chunks.document_id` FK can be satisfied by `persist_table_chunks`.
///
/// The pipeline-side upload writes here implicitly via text_upload.rs;
/// the lattice fast-path bypasses that and writes only to `kv_storage`,
/// which leaves `documents` empty and breaks any feature that resolves
/// docs through the relational layer (chunks FK, the /documents/{id}
/// detail page's `chunks` lookups, future SQL-side searches).
///
/// Only the schema-required columns are populated:
/// - `id`           the UUID we generated
/// - `tenant_id`    + `workspace_id` (from the request context)
/// - `title`        from req.title, fallback to source_id, fallback to a
///                  generated label
/// - `content`      the raw markdown body
/// - `content_hash` SHA-256 of content
/// - `status`       lattice's initial status
/// - `file_path`    req.source_id (relative path)
/// - `metadata`     `{ "source_type": "markdown", "managed_by": "lattice" }`
#[cfg(feature = "postgres")]
async fn upsert_document_row(
    state: &AppState,
    document_id: &str,
    req: &crate::handlers::documents_types::RawDocumentRequest,
    workspace_id: &str,
    tenant_id: &Option<String>,
    status: &str,
    content_hash: &str,
) -> Result<(), String> {
    let Some(pool) = state.pg_pool.as_ref() else {
        return Err("Postgres pool not configured".to_string());
    };
    let doc_uuid = Uuid::parse_str(document_id).map_err(|e| format!("invalid doc_id: {e}"))?;
    let workspace_uuid =
        Uuid::parse_str(workspace_id).map_err(|e| format!("invalid workspace_id: {e}"))?;
    let tenant_uuid = tenant_id
        .as_ref()
        .and_then(|s| Uuid::parse_str(s).ok())
        .ok_or_else(|| "tenant_id is not a UUID".to_string())?;
    let title = req
        .title
        .clone()
        .or_else(|| req.source_id.clone())
        .unwrap_or_else(|| format!("doc-{}", document_id));
    let metadata = serde_json::json!({
        "source_type": "markdown",
        "managed_by": "lattice",
    });

    sqlx::query(
        r#"
        INSERT INTO documents (
            id, tenant_id, workspace_id, title, content, content_hash,
            status, file_path, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (id) DO UPDATE SET
            title         = EXCLUDED.title,
            content       = EXCLUDED.content,
            content_hash  = EXCLUDED.content_hash,
            status        = EXCLUDED.status,
            file_path     = EXCLUDED.file_path,
            metadata      = EXCLUDED.metadata,
            updated_at    = now()
        "#,
    )
    .bind(doc_uuid)
    .bind(tenant_uuid)
    .bind(workspace_uuid)
    .bind(&title)
    .bind(&req.content)
    .bind(content_hash)
    .bind(status)
    .bind(&req.source_id)
    .bind(&metadata)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Mirror table-kind chunks into the `chunks` SQL table so they appear on
/// the Figures tab. Markdown tables come from lattice's chunker as plain
/// chunk uploads with `metadata.element_kind == "table"`; we parse the
/// markdown table inline, compute a basic HTML rendering + rows JSON, and
/// upsert one row per table.
///
/// Failures are logged at warn level and don't fail the chunks call —
/// table mirroring is a UX feature, not a correctness invariant.
#[cfg(feature = "postgres")]
async fn persist_table_chunks(
    state: &AppState,
    doc_id: &str,
    tenant_id: &Option<String>,
    workspace_id: &str,
    chunks: &[crate::handlers::documents_types::ChunkUpload],
) -> Result<(), String> {
    let table_chunks: Vec<&crate::handlers::documents_types::ChunkUpload> = chunks
        .iter()
        .filter(|c| {
            c.metadata
                .get("element_kind")
                .and_then(|v| v.as_str())
                .is_some_and(|s| s == "table")
        })
        .collect();
    if table_chunks.is_empty() {
        return Ok(());
    }

    let Some(pool) = state.pg_pool.as_ref() else {
        return Err("Postgres pool not configured".to_string());
    };
    let doc_uuid = Uuid::parse_str(doc_id).map_err(|e| format!("invalid doc_id: {e}"))?;
    let tenant_uuid = tenant_id
        .as_ref()
        .and_then(|s| Uuid::parse_str(s).ok())
        .ok_or_else(|| "tenant_id is not a UUID".to_string())?;
    let workspace_uuid =
        Uuid::parse_str(workspace_id).map_err(|e| format!("invalid workspace_id: {e}"))?;

    // chunk_index space: pipeline-side tables use 2_000_000+ to stay
    // disjoint from text (≤~1M) and figures (1M-2M). Mirror that by
    // offsetting lattice's chunk_index into the same range.
    const TABLE_CHUNK_INDEX_BASE: i32 = 2_000_000;

    for c in table_chunks {
        let chunk_index = c
            .metadata
            .get("chunk_index")
            .and_then(|v| v.as_i64())
            .map(|n| TABLE_CHUNK_INDEX_BASE + (n as i32))
            .unwrap_or(TABLE_CHUNK_INDEX_BASE);

        // Synthesize a table_id: prefer lattice's chunk id, fall back to
        // a deterministic synth so detail-page links stay stable across
        // syncs.
        let table_id = c.id.clone();
        let (table_html, table_rows) = markdown_table_to_html_and_rows(&c.content);

        // The Figures-tab list endpoint reads `chunks.content` and surfaces
        // it as `caption`. If lattice supplied a generated caption in the
        // chunk's metadata, store that in `content`; the raw markdown lives
        // on in `table_html` + `table_rows` for the detail view. Without a
        // caption (PR1 heuristic miss, PR2 LLM not yet run), fall back to
        // the raw markdown so the column still has something readable.
        let caption = c
            .metadata
            .get("caption")
            .and_then(|v| v.as_str())
            .map(|s| s.trim())
            .filter(|s| !s.is_empty());
        let content_for_chunks_row: &str = caption.unwrap_or(c.content.as_str());

        let order_index = c
            .metadata
            .get("chunk_index")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let table_number = c.metadata.get("table_number").and_then(|v| v.as_i64());
        let table_metadata = serde_json::json!({
            "kind": "table",
            "table_id": &table_id,
            "page": 1,
            "order_index": order_index,
            "source": "lattice",
            "caption": caption,
            "table_number": table_number,
            "raw_md": &c.content,
        });

        let res = sqlx::query(
            r#"
            INSERT INTO chunks (
                document_id, tenant_id, workspace_id,
                content, chunk_index,
                kind, table_id, table_html, table_rows,
                metadata
            ) VALUES ($1, $2, $3, $4, $5, 'table', $6, $7, $8, $9)
            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                content    = EXCLUDED.content,
                kind       = EXCLUDED.kind,
                table_id   = EXCLUDED.table_id,
                table_html = EXCLUDED.table_html,
                table_rows = EXCLUDED.table_rows,
                metadata   = EXCLUDED.metadata
            "#,
        )
        .bind(doc_uuid)
        .bind(tenant_uuid)
        .bind(workspace_uuid)
        .bind(content_for_chunks_row)
        .bind(chunk_index)
        .bind(&table_id)
        .bind(&table_html)
        .bind(&table_rows)
        .bind(&table_metadata)
        .execute(pool)
        .await;
        if let Err(e) = res {
            warn!(
                table_id = %table_id,
                document_id = %doc_id,
                error = %e,
                "lattice table mirror: row insert failed"
            );
        }
    }

    Ok(())
}

/// Minimal markdown-table → (HTML, parsed rows JSON) converter. Handles
/// the GitHub-flavored markdown table shape lattice's chunker emits:
///
/// ```text
/// | col1 | col2 |
/// |------|------|
/// | a    | b    |
/// ```
///
/// Anything that doesn't parse as a table is returned as a single-row
/// "table" with the raw content — the Figures tab UI can still display
/// it via the caption field.
#[cfg(feature = "postgres")]
fn markdown_table_to_html_and_rows(md: &str) -> (String, serde_json::Value) {
    let mut lines: Vec<&str> = md.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return (String::new(), serde_json::json!({ "headers": [], "rows": [] }));
    }

    let split_cells = |line: &str| -> Vec<String> {
        let trimmed = line.trim();
        let inner = trimmed
            .strip_prefix('|')
            .map(|s| s.strip_suffix('|').unwrap_or(s))
            .unwrap_or(trimmed);
        inner
            .split('|')
            .map(|c| c.trim().to_string())
            .collect()
    };
    let is_separator = |line: &str| -> bool {
        let cells = split_cells(line);
        !cells.is_empty()
            && cells
                .iter()
                .all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' '))
    };

    let headers: Vec<String>;
    let body_lines: Vec<&str>;
    if lines.len() >= 2 && is_separator(lines[1]) {
        headers = split_cells(lines[0]);
        body_lines = lines.drain(2..).collect();
    } else {
        // No separator — treat as headerless content; first line is row 0.
        headers = Vec::new();
        body_lines = lines.drain(..).collect();
    }

    let rows: Vec<Vec<String>> = body_lines.into_iter().map(split_cells).collect();

    let mut html = String::from("<table border=\"1\">");
    if !headers.is_empty() {
        html.push_str("<thead><tr>");
        for h in &headers {
            html.push_str(&format!("<th>{}</th>", html_escape(h)));
        }
        html.push_str("</tr></thead>");
    }
    html.push_str("<tbody>");
    for row in &rows {
        html.push_str("<tr>");
        for cell in row {
            html.push_str(&format!("<td>{}</td>", html_escape(cell)));
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table>");

    let rows_json = serde_json::json!({
        "headers": headers,
        "rows": rows,
    });
    (html, rows_json)
}

#[cfg(feature = "postgres")]
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Reject the request if the doc's stored tenant/workspace doesn't match
/// the caller's TenantContext. Mirrors detail.rs's check.
fn enforce_tenant_match(
    metadata: &serde_json::Value,
    tenant_ctx: &TenantContext,
) -> ApiResult<()> {
    let obj = match metadata.as_object() {
        Some(o) => o,
        None => return Ok(()),
    };
    let doc_tenant = obj.get("tenant_id").and_then(|v| v.as_str());
    let doc_workspace = obj.get("workspace_id").and_then(|v| v.as_str());

    if let Some(ref ctx_tid) = tenant_ctx.tenant_id {
        if let Some(doc_tid) = doc_tenant {
            if doc_tid != ctx_tid {
                return Err(ApiError::Forbidden);
            }
        }
    }
    if let Some(ref ctx_ws) = tenant_ctx.workspace_id {
        if let Some(doc_ws) = doc_workspace {
            if doc_ws != ctx_ws {
                return Err(ApiError::Forbidden);
            }
        }
    }
    Ok(())
}
