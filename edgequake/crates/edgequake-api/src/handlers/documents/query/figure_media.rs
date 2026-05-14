//! Figure media-fetch endpoint.
//!
//! Streams the raw PNG bytes of a captured figure back to clients so the UI
//! can render it next to retrieval hits. Figure bytes are persisted by the
//! VLM-OCR pipeline into the `chunks` table (`kind = 'figure'`, `figure_id`
//! set, `media_bytes` non-null) — see `migrations/052_add_media_to_chunks.sql`
//! and the backfill in `processor/pdf_processing.rs::backfill_figure_media`.

use axum::extract::{Path, State};

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

/// `GET /api/v1/documents/{document_id}/figures/{figure_id}`
///
/// Returns the PNG bytes of a figure cropped during VLM-OCR conversion, with
/// the persisted MIME as the content-type. Workspace isolation is enforced:
/// the chunk's workspace_id must match the caller's context (when provided).
///
/// 404 when no figure row matches `(document_id, figure_id)` or when
/// `media_bytes` is NULL on the row.
#[utoipa::path(
    get,
    path = "/api/v1/documents/{document_id}/figures/{figure_id}",
    params(
        ("document_id" = String, Path, description = "Document identifier (UUID)"),
        ("figure_id" = String, Path, description = "Extractor-side figure id, e.g. `fig_3_5`")
    ),
    responses(
        (status = 200, description = "Raw figure bytes", content_type = "image/png"),
        (status = 404, description = "Figure not found"),
        (status = 403, description = "Not authorized for this workspace"),
        (status = 500, description = "Internal server error")
    ),
    tag = "Documents"
)]
pub async fn get_figure_media(
    State(state): State<AppState>,
    context: TenantContext,
    Path((document_id, figure_id)): Path<(String, String)>,
) -> ApiResult<axum::response::Response<axum::body::Body>> {
    use axum::http::header;
    use axum::response::IntoResponse;

    let doc_uuid = uuid::Uuid::parse_str(&document_id)
        .map_err(|_| ApiError::BadRequest("Invalid document_id (expected UUID)".to_string()))?;

    #[cfg(feature = "postgres")]
    {
        let pool = state
            .pg_pool
            .as_ref()
            .ok_or_else(|| ApiError::Internal("Postgres pool not configured".to_string()))?;

        let row: Option<(Vec<u8>, Option<String>, Option<uuid::Uuid>)> = sqlx::query_as(
            r#"
            SELECT media_bytes, media_mime, workspace_id
            FROM chunks
            WHERE document_id = $1
              AND figure_id   = $2
              AND kind        = 'figure'
              AND media_bytes IS NOT NULL
            LIMIT 1
            "#,
        )
        .bind(doc_uuid)
        .bind(&figure_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| ApiError::Internal(format!("figure media query failed: {e}")))?;

        let (bytes, mime, ws) =
            row.ok_or_else(|| ApiError::NotFound("Figure not found".to_string()))?;

        // Workspace isolation: only check when the caller supplied a header AND
        // the row has a workspace. Mirrors `download_pdf`'s defense-in-depth
        // model — figures inherit their owner workspace from the chunk row.
        if let (Some(caller_ws), Some(row_ws)) = (context.workspace_id_uuid(), ws) {
            if caller_ws != row_ws {
                return Err(ApiError::Forbidden);
            }
        }

        let content_type = mime.unwrap_or_else(|| "image/png".to_string());
        Ok((
            [
                (header::CONTENT_TYPE, content_type),
                (
                    header::CACHE_CONTROL,
                    "private, max-age=3600".to_string(),
                ),
            ],
            bytes,
        )
            .into_response())
    }

    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, context, doc_uuid, figure_id);
        Err(ApiError::Internal(
            "Figure media endpoint requires the postgres feature".to_string(),
        ))
    }
}
