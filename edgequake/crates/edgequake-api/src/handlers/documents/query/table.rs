//! Table-fetch endpoints.
//!
//! Two handlers:
//!
//! - `GET /api/v1/documents/{document_id}/tables` — light list payload for
//!   the gallery (one row per table, no HTML / rows).
//! - `GET /api/v1/documents/{document_id}/tables/{table_id}` — full payload
//!   for inline rendering and detail view (HTML + parsed `{headers, rows}` +
//!   classification + caption).
//!
//! Both enforce workspace isolation the same way `figure_media.rs` does:
//! when the caller supplies a workspace context AND the row has a
//! workspace, mismatch returns 403.
//!
//! Companion endpoint `list_document_figures` lives here too so the
//! frontend gallery can fetch both in parallel from the same module.

use axum::extract::{Path, State};
use axum::Json;
use serde::Serialize;

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

/// Light list-row shape for the gallery / type-filter UI. No HTML / rows —
/// those land on the detail endpoint instead so list payloads stay small
/// even for documents with dozens of large tables.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentTableListItem {
    pub table_id: String,
    pub caption: String,
    pub page: u32,
    pub order_index: u32,
    /// One of `performance` / `quality` / `complexity` / `other`. `None`
    /// while the classification stage hasn't run yet; the gallery should
    /// treat that as "unclassified" rather than hiding the row.
    pub table_type: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DocumentTableListResponse {
    pub document_id: String,
    pub tables: Vec<DocumentTableListItem>,
}

/// Full detail payload returned by the by-id endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentTableDetailResponse {
    pub table_id: String,
    pub caption: String,
    pub page: u32,
    pub order_index: u32,
    pub table_type: Option<String>,
    pub rationale: Option<String>,
    /// VLM-rendered HTML for inline display.
    pub html: String,
    /// Algorithmically parsed structure: `{ "headers": [...], "rows": [[...], ...] }`.
    /// May be `null` if the parser couldn't decode the HTML.
    pub rows: serde_json::Value,
}

/// Light list-row shape for the figures gallery — mirrors the table list
/// shape so the frontend can interleave figures and tables in one grid.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentFigureListItem {
    pub figure_id: String,
    pub caption: String,
    pub page: u32,
    pub order_index: u32,
    pub mime: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DocumentFigureListResponse {
    pub document_id: String,
    pub figures: Vec<DocumentFigureListItem>,
}

/// `GET /api/v1/documents/{document_id}/tables`
#[utoipa::path(
    get,
    path = "/api/v1/documents/{document_id}/tables",
    params(
        ("document_id" = String, Path, description = "Document identifier (UUID)")
    ),
    responses(
        (status = 200, description = "List of tables for the document"),
        (status = 400, description = "Invalid document_id"),
        (status = 500, description = "Internal server error")
    ),
    tag = "Documents"
)]
pub async fn list_document_tables(
    State(state): State<AppState>,
    context: TenantContext,
    Path(document_id): Path<String>,
) -> ApiResult<Json<DocumentTableListResponse>> {
    let doc_uuid = uuid::Uuid::parse_str(&document_id)
        .map_err(|_| ApiError::BadRequest("Invalid document_id (expected UUID)".to_string()))?;

    #[cfg(feature = "postgres")]
    {
        let pool = state
            .pg_pool
            .as_ref()
            .ok_or_else(|| ApiError::Internal("Postgres pool not configured".to_string()))?;

        // Caller's workspace gates the result set so a user in one workspace
        // can't enumerate another workspace's tables by guessing document
        // ids. Mirrors the row-level isolation `figure_media.rs` applies on
        // the by-id endpoint.
        let caller_ws = context.workspace_id_uuid();

        let rows: Vec<(
            Option<String>,
            Option<String>,
            Option<serde_json::Value>,
            Option<String>,
            Option<uuid::Uuid>,
        )> = sqlx::query_as(
            r#"
            SELECT
                table_id,
                content,
                metadata,
                table_type,
                workspace_id
            FROM chunks
            WHERE document_id = $1
              AND kind = 'table'
              AND table_id IS NOT NULL
            ORDER BY chunk_index
            "#,
        )
        .bind(doc_uuid)
        .fetch_all(pool)
        .await
        .map_err(|e| ApiError::Internal(format!("table list query failed: {e}")))?;

        let mut items: Vec<DocumentTableListItem> = Vec::with_capacity(rows.len());
        for (table_id, content, metadata, table_type, ws) in rows {
            if let (Some(caller), Some(row_ws)) = (caller_ws, ws) {
                if caller != row_ws {
                    continue;
                }
            }
            let Some(id) = table_id else { continue };
            let caption = content.unwrap_or_default();
            let (page, order_index) = page_order_from_metadata(metadata.as_ref());
            items.push(DocumentTableListItem {
                table_id: id,
                caption,
                page,
                order_index,
                table_type,
            });
        }

        Ok(Json(DocumentTableListResponse {
            document_id,
            tables: items,
        }))
    }

    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, context, doc_uuid);
        Err(ApiError::Internal(
            "Table list endpoint requires the postgres feature".to_string(),
        ))
    }
}

/// `GET /api/v1/documents/{document_id}/tables/{table_id}`
#[utoipa::path(
    get,
    path = "/api/v1/documents/{document_id}/tables/{table_id}",
    params(
        ("document_id" = String, Path, description = "Document identifier (UUID)"),
        ("table_id" = String, Path, description = "Extractor-side table id, e.g. `tbl_3_5`")
    ),
    responses(
        (status = 200, description = "Full table payload"),
        (status = 404, description = "Table not found"),
        (status = 403, description = "Not authorized for this workspace"),
        (status = 500, description = "Internal server error")
    ),
    tag = "Documents"
)]
pub async fn get_document_table(
    State(state): State<AppState>,
    context: TenantContext,
    Path((document_id, table_id)): Path<(String, String)>,
) -> ApiResult<Json<DocumentTableDetailResponse>> {
    let doc_uuid = uuid::Uuid::parse_str(&document_id)
        .map_err(|_| ApiError::BadRequest("Invalid document_id (expected UUID)".to_string()))?;

    #[cfg(feature = "postgres")]
    {
        let pool = state
            .pg_pool
            .as_ref()
            .ok_or_else(|| ApiError::Internal("Postgres pool not configured".to_string()))?;

        let row: Option<(
            Option<String>,
            Option<String>,
            Option<serde_json::Value>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<serde_json::Value>,
            Option<uuid::Uuid>,
        )> = sqlx::query_as(
            r#"
            SELECT
                content,
                table_html,
                table_rows,
                table_type,
                table_classification_rationale,
                table_id,
                metadata,
                workspace_id
            FROM chunks
            WHERE document_id = $1
              AND table_id   = $2
              AND kind       = 'table'
            LIMIT 1
            "#,
        )
        .bind(doc_uuid)
        .bind(&table_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| ApiError::Internal(format!("table fetch query failed: {e}")))?;

        let (content, html, rows_json, table_type, rationale, table_id_db, metadata, ws) =
            row.ok_or_else(|| ApiError::NotFound("Table not found".to_string()))?;

        if let (Some(caller_ws), Some(row_ws)) = (context.workspace_id_uuid(), ws) {
            if caller_ws != row_ws {
                return Err(ApiError::Forbidden);
            }
        }

        let (page, order_index) = page_order_from_metadata(metadata.as_ref());
        Ok(Json(DocumentTableDetailResponse {
            table_id: table_id_db.unwrap_or(table_id),
            caption: content.unwrap_or_default(),
            page,
            order_index,
            table_type,
            rationale,
            html: html.unwrap_or_default(),
            rows: rows_json.unwrap_or(serde_json::Value::Null),
        }))
    }

    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, context, doc_uuid, table_id);
        Err(ApiError::Internal(
            "Table detail endpoint requires the postgres feature".to_string(),
        ))
    }
}

/// `GET /api/v1/documents/{document_id}/figures`
///
/// Sibling to `list_document_tables` so the gallery can pull both lists
/// without screen-scraping the markdown for figure sentinels. The bytes
/// still live behind `get_figure_media` — this endpoint just enumerates.
#[utoipa::path(
    get,
    path = "/api/v1/documents/{document_id}/figures",
    params(
        ("document_id" = String, Path, description = "Document identifier (UUID)")
    ),
    responses(
        (status = 200, description = "List of figures for the document"),
        (status = 400, description = "Invalid document_id"),
        (status = 500, description = "Internal server error")
    ),
    tag = "Documents"
)]
pub async fn list_document_figures(
    State(state): State<AppState>,
    context: TenantContext,
    Path(document_id): Path<String>,
) -> ApiResult<Json<DocumentFigureListResponse>> {
    let doc_uuid = uuid::Uuid::parse_str(&document_id)
        .map_err(|_| ApiError::BadRequest("Invalid document_id (expected UUID)".to_string()))?;

    #[cfg(feature = "postgres")]
    {
        let pool = state
            .pg_pool
            .as_ref()
            .ok_or_else(|| ApiError::Internal("Postgres pool not configured".to_string()))?;

        let caller_ws = context.workspace_id_uuid();

        let rows: Vec<(
            Option<String>,
            Option<String>,
            Option<serde_json::Value>,
            Option<String>,
            Option<uuid::Uuid>,
        )> = sqlx::query_as(
            r#"
            SELECT
                figure_id,
                content,
                metadata,
                media_mime,
                workspace_id
            FROM chunks
            WHERE document_id = $1
              AND kind = 'figure'
              AND figure_id IS NOT NULL
            ORDER BY chunk_index
            "#,
        )
        .bind(doc_uuid)
        .fetch_all(pool)
        .await
        .map_err(|e| ApiError::Internal(format!("figure list query failed: {e}")))?;

        let mut items: Vec<DocumentFigureListItem> = Vec::with_capacity(rows.len());
        for (figure_id, content, metadata, mime, ws) in rows {
            if let (Some(caller), Some(row_ws)) = (caller_ws, ws) {
                if caller != row_ws {
                    continue;
                }
            }
            let Some(id) = figure_id else { continue };
            let caption = content.unwrap_or_default();
            let (page, order_index) = page_order_from_metadata(metadata.as_ref());
            items.push(DocumentFigureListItem {
                figure_id: id,
                caption,
                page,
                order_index,
                mime,
            });
        }

        Ok(Json(DocumentFigureListResponse {
            document_id,
            figures: items,
        }))
    }

    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, context, doc_uuid);
        Err(ApiError::Internal(
            "Figure list endpoint requires the postgres feature".to_string(),
        ))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ReclassifyTablesResponse {
    pub document_id: String,
    pub tables_reset: u64,
    pub track_id: String,
}

/// `POST /api/v1/documents/{document_id}/tables/reclassify`
///
/// Resets `table_type` + `table_classification_rationale` to NULL for every
/// table row of this document, then enqueues a `TaskType::TableClassification`
/// task that re-runs the LLM classifier. Useful when the inline classifier
/// got cut off mid-document, when the workspace LLM was swapped, or when
/// the operator wants to re-label all rows after iterating on the prompt.
///
/// Returns the count of rows reset and the track id of the enqueued task so
/// the frontend can poll for progress.
#[utoipa::path(
    post,
    path = "/api/v1/documents/{document_id}/tables/reclassify",
    params(
        ("document_id" = String, Path, description = "Document identifier (UUID)")
    ),
    responses(
        (status = 200, description = "Reclassification task enqueued"),
        (status = 400, description = "Invalid document_id"),
        (status = 500, description = "Internal server error")
    ),
    tag = "Documents"
)]
pub async fn reclassify_document_tables(
    State(state): State<AppState>,
    context: TenantContext,
    Path(document_id): Path<String>,
) -> ApiResult<Json<ReclassifyTablesResponse>> {
    let doc_uuid = uuid::Uuid::parse_str(&document_id)
        .map_err(|_| ApiError::BadRequest("Invalid document_id (expected UUID)".to_string()))?;

    #[cfg(feature = "postgres")]
    {
        use edgequake_tasks::{Task, TableClassificationData, TaskType};

        let pool = state
            .pg_pool
            .as_ref()
            .ok_or_else(|| ApiError::Internal("Postgres pool not configured".to_string()))?;

        // Resolve the workspace + tenant the table rows belong to. We trust
        // the caller's TenantContext when set, but also pull from the
        // chunks row as a defense-in-depth check — refuse the request if
        // the caller's workspace doesn't match the chunk's workspace.
        let row: Option<(Option<uuid::Uuid>, Option<uuid::Uuid>)> = sqlx::query_as(
            r#"
            SELECT tenant_id, workspace_id
            FROM chunks
            WHERE document_id = $1 AND kind = 'table'
            LIMIT 1
            "#,
        )
        .bind(doc_uuid)
        .fetch_optional(pool)
        .await
        .map_err(|e| ApiError::Internal(format!("table workspace lookup failed: {e}")))?;

        let (row_tenant, row_workspace) =
            row.ok_or_else(|| ApiError::NotFound("No tables for document".to_string()))?;
        let row_workspace = row_workspace
            .ok_or_else(|| ApiError::Internal("Table row missing workspace_id".to_string()))?;
        let row_tenant = row_tenant
            .ok_or_else(|| ApiError::Internal("Table row missing tenant_id".to_string()))?;

        if let Some(caller_ws) = context.workspace_id_uuid() {
            if caller_ws != row_workspace {
                return Err(ApiError::Forbidden);
            }
        }

        // Reset existing classifications so the inline classifier's
        // `WHERE table_type IS NULL` filter picks every row up.
        let reset = sqlx::query(
            r#"
            UPDATE chunks
            SET table_type = NULL,
                table_classification_rationale = NULL
            WHERE document_id = $1 AND kind = 'table'
            "#,
        )
        .bind(doc_uuid)
        .execute(pool)
        .await
        .map_err(|e| ApiError::Internal(format!("table reset failed: {e}")))?;

        // Enqueue the classification task. The worker picks it up the same
        // way it picks up the inline-fired one from `process_pdf_processing`.
        let data = TableClassificationData {
            document_id: document_id.clone(),
            workspace_id: row_workspace.to_string(),
        };
        let task = Task::new(
            row_tenant,
            row_workspace,
            TaskType::TableClassification,
            serde_json::to_value(data).map_err(|e| {
                ApiError::Internal(format!("Failed to serialize task data: {e}"))
            })?,
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

        Ok(Json(ReclassifyTablesResponse {
            document_id,
            tables_reset: reset.rows_affected(),
            track_id,
        }))
    }

    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, context, doc_uuid);
        Err(ApiError::Internal(
            "Table reclassify endpoint requires the postgres feature".to_string(),
        ))
    }
}

/// Best-effort `(page, order_index)` extraction from the chunk's metadata
/// JSON (written by `backfill_table_content` / `backfill_figure_media`).
/// Returns `(0, 0)` if either field is missing or non-numeric — the gallery
/// still renders, it just loses precise ordering for that row.
fn page_order_from_metadata(metadata: Option<&serde_json::Value>) -> (u32, u32) {
    let Some(m) = metadata else {
        return (0, 0);
    };
    let page = m
        .get("page")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(0);
    let order_index = m
        .get("order_index")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(0);
    (page, order_index)
}
