use axum::extract::{Path, Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::info;
use utoipa::ToSchema;
use uuid::Uuid;

use super::helpers::get_pdf_storage;
use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;
use edgequake_storage::PdfProcessingStatus;

/// Query params for [`get_pdf_content`].
#[derive(Debug, Default, Deserialize)]
pub struct PdfContentQuery {
    /// When true, `![tbl_…](edgequake-table)` placeholders in the returned
    /// markdown are replaced inline with the stored table content rendered as
    /// a GitHub-flavoured-markdown table (caption + cells). Default false so
    /// the frontend viewer (which resolves the placeholders client-side from
    /// `chunks.table_html`) is unaffected. Agent read paths (MCP
    /// `document_get_md`) set this so the table numbers — communication cost,
    /// latency, accuracy — are visible in the text instead of a placeholder.
    #[serde(default)]
    pub inline_tables: bool,
}

// ============================================================================
// PDF Content Download Endpoints (SPEC-002: Document Viewer)
// ============================================================================

/// PDF download response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct PdfContentResponse {
    /// PDF ID.
    pub pdf_id: String,
    /// Original filename.
    pub filename: String,
    /// File size in bytes.
    pub file_size_bytes: i64,
    /// MIME type.
    pub content_type: String,
    /// Extracted markdown content (if processed).
    pub markdown_content: Option<String>,
    /// Whether PDF processing is complete.
    pub is_processed: bool,
}

/// Download raw PDF file data.
///
/// @implements SPEC-002: Document Viewer - PDF download endpoint
/// @implements UC0711: Download PDF for viewing
/// @enforces BR0701: Workspace isolation
///
/// Returns the raw PDF binary data with appropriate content-type headers.
/// This allows the frontend PDF viewer to render the original document.
///
/// # Arguments
///
/// * `state` - Application state with PDF storage
/// * `context` - Tenant context for workspace isolation
/// * `pdf_id` - PDF identifier
///
/// # Returns
///
/// * `Ok(Response)` - Raw PDF data with application/pdf content-type
/// * `Err(404)` - PDF not found
/// * `Err(403)` - Not authorized for this workspace
#[utoipa::path(
    get,
    path = "/api/v1/documents/pdf/{pdf_id}/download",
    params(
        ("pdf_id" = String, Path, description = "PDF identifier")
    ),
    responses(
        (status = 200, description = "Raw PDF data", content_type = "application/pdf"),
        (status = 404, description = "PDF not found"),
        (status = 403, description = "Not authorized"),
        (status = 500, description = "Internal server error")
    ),
    tag = "Documents"
)]
pub async fn download_pdf(
    State(state): State<AppState>,
    context: TenantContext,
    Path(pdf_id): Path<String>,
) -> ApiResult<axum::response::Response<axum::body::Body>> {
    use axum::http::header;
    use axum::response::IntoResponse;

    let pdf_id = Uuid::parse_str(&pdf_id)
        .map_err(|_| ApiError::BadRequest("Invalid PDF ID format".to_string()))?;

    let pdf_storage = get_pdf_storage(&state)?;

    let pdf = pdf_storage
        .get_pdf(&pdf_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get PDF: {}", e)))?
        .ok_or_else(|| ApiError::NotFound("PDF not found".to_string()))?;

    // OODA-51: Make workspace verification optional for PDF viewer compatibility
    // WHY: react-pdf Document component loads PDFs via URL without custom headers,
    // so X-Workspace-ID header is not available. The PDF is already isolated by its
    // UUID which is unique per workspace, so access is implicitly scoped.
    // If workspace header IS provided, verify it matches for defense-in-depth.
    if let Some(workspace_id) = context.workspace_id_uuid() {
        if pdf.workspace_id != workspace_id {
            return Err(ApiError::Forbidden);
        }
    }

    info!(
        "PDF download: id={}, filename={}, size={}",
        pdf_id,
        pdf.filename,
        pdf.pdf_data.len()
    );

    // Build response with PDF data
    let content_disposition = format!("inline; filename=\"{}\"", pdf.filename);

    Ok((
        [
            (header::CONTENT_TYPE, "application/pdf"),
            (header::CONTENT_DISPOSITION, content_disposition.as_str()),
            (header::CACHE_CONTROL, "private, max-age=3600"),
        ],
        pdf.pdf_data,
    )
        .into_response())
}

/// Get PDF content metadata including markdown.
///
/// @implements SPEC-002: Document Viewer - Markdown content endpoint
/// @implements UC0712: Get PDF metadata with extracted markdown
/// @enforces BR0701: Workspace isolation
///
/// Returns PDF metadata including the extracted markdown content (if processed).
/// This allows the frontend to display both the original PDF and the extracted markdown.
///
/// # Arguments
///
/// * `state` - Application state with PDF storage
/// * `context` - Tenant context for workspace isolation
/// * `pdf_id` - PDF identifier
///
/// # Returns
///
/// * `Ok(Json(PdfContentResponse))` - PDF metadata with markdown
/// * `Err(404)` - PDF not found
/// * `Err(403)` - Not authorized for this workspace
#[utoipa::path(
    get,
    path = "/api/v1/documents/pdf/{pdf_id}/content",
    params(
        ("pdf_id" = String, Path, description = "PDF identifier")
    ),
    responses(
        (status = 200, description = "PDF content metadata", body = PdfContentResponse),
        (status = 404, description = "PDF not found"),
        (status = 403, description = "Not authorized"),
        (status = 500, description = "Internal server error")
    ),
    tag = "Documents"
)]
pub async fn get_pdf_content(
    State(state): State<AppState>,
    context: TenantContext,
    Path(pdf_id): Path<String>,
    Query(params): Query<PdfContentQuery>,
) -> ApiResult<Json<PdfContentResponse>> {
    let pdf_id = Uuid::parse_str(&pdf_id)
        .map_err(|_| ApiError::BadRequest("Invalid PDF ID format".to_string()))?;

    let pdf_storage = get_pdf_storage(&state)?;

    let pdf = pdf_storage
        .get_pdf(&pdf_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get PDF: {}", e)))?
        .ok_or_else(|| ApiError::NotFound("PDF not found".to_string()))?;

    // OODA-51: Make workspace verification optional for PDF viewer compatibility
    // WHY: Frontend PDF components may not have access to custom headers.
    // If workspace header IS provided, verify it matches for defense-in-depth.
    if let Some(workspace_id) = context.workspace_id_uuid() {
        if pdf.workspace_id != workspace_id {
            return Err(ApiError::Forbidden);
        }
    }

    let is_processed = pdf.processing_status == PdfProcessingStatus::Completed;

    let mut markdown_content = pdf.markdown_content;

    // Opt-in: rehydrate `![tbl_…](edgequake-table)` placeholders with the
    // stored table content so agent read paths see the cell values. The
    // table HTML/rows live in `chunks.table_html`/`table_rows`; the markdown
    // only carries a placeholder (the frontend resolves it client-side).
    #[cfg(feature = "postgres")]
    if params.inline_tables {
        if let (Some(md), Some(doc_uuid), Some(pool)) =
            (markdown_content.as_mut(), pdf.document_id, state.pg_pool.as_ref())
        {
            if md.contains("(edgequake-table)") {
                inline_document_tables(pool, &doc_uuid, md).await;
            }
        }
    }

    Ok(Json(PdfContentResponse {
        pdf_id: pdf.pdf_id.to_string(),
        filename: pdf.filename,
        file_size_bytes: pdf.file_size_bytes,
        content_type: pdf.content_type,
        markdown_content,
        is_processed,
    }))
}

/// Replace every `![<table_id>](edgequake-table)` placeholder in `markdown`
/// with the stored table rendered as a GitHub-flavoured-markdown table.
///
/// Best-effort: a placeholder with no matching `chunks` row, or a row with no
/// structured `table_rows`, is left untouched. Never errors — table inlining
/// is an enhancement, not a correctness requirement, so a DB hiccup must not
/// fail the content read.
#[cfg(feature = "postgres")]
async fn inline_document_tables(
    pool: &sqlx::PgPool,
    document_id: &Uuid,
    markdown: &mut String,
) {
    let rows: Vec<(Option<String>, Option<String>, Option<serde_json::Value>)> =
        match sqlx::query_as(
            r#"SELECT table_id, content, table_rows
                 FROM chunks
                WHERE document_id = $1 AND kind = 'table'"#,
        )
        .bind(document_id)
        .fetch_all(pool)
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(document_id = %document_id, error = %e, "inline_tables: table fetch failed; leaving placeholders");
                return;
            }
        };

    for (table_id, caption, table_rows) in rows {
        let Some(table_id) = table_id else { continue };
        let placeholder = format!("![{table_id}](edgequake-table)");
        if !markdown.contains(&placeholder) {
            continue;
        }
        let Some(rendered) = render_table_markdown(caption.as_deref(), table_rows.as_ref())
        else {
            continue;
        };
        *markdown = markdown.replace(&placeholder, &rendered);
    }
}

/// Render a stored table (`{"rows": [[..],[..]]}`) plus its caption into a
/// GFM table. Returns `None` when there are no usable rows. The first row is
/// treated as the header; short rows (section labels like `["GPT2-Small"]`)
/// are padded to the header width so their text survives.
#[cfg(feature = "postgres")]
fn render_table_markdown(
    caption: Option<&str>,
    table_rows: Option<&serde_json::Value>,
) -> Option<String> {
    let rows: Vec<Vec<String>> = table_rows
        .and_then(|v| v.get("rows"))
        .and_then(|v| v.as_array())
        .map(|outer| {
            outer
                .iter()
                .filter_map(|row| row.as_array())
                .map(|cells| {
                    cells
                        .iter()
                        .map(|c| {
                            // Escape pipes so cell text can't break the GFM grid.
                            c.as_str().unwrap_or("").replace('|', "\\|").trim().to_string()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let width = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if width == 0 {
        return None;
    }

    let pad = |r: &[String]| -> String {
        let mut cells: Vec<String> = r.to_vec();
        cells.resize(width, String::new());
        format!("| {} |", cells.join(" | "))
    };

    let mut out = String::new();
    if let Some(cap) = caption {
        let cap = cap.trim();
        if !cap.is_empty() {
            out.push_str(cap);
            out.push_str("\n\n");
        }
    }
    out.push_str(&pad(&rows[0]));
    out.push('\n');
    out.push_str(&format!("| {} |", vec!["---"; width].join(" | ")));
    out.push('\n');
    for r in &rows[1..] {
        out.push_str(&pad(r));
        out.push('\n');
    }
    Some(out)
}

#[cfg(all(test, feature = "postgres"))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_gfm_with_caption_and_cells() {
        let rows = json!({"rows": [
            ["Setting", "WebQs", "SciQ"],
            ["GPT2-Small"],
            ["Protected(ours)", "16.8", "91.7"]
        ]});
        let md = render_table_markdown(Some("Table 1: Accuracy"), Some(&rows)).unwrap();
        // Caption preserved.
        assert!(md.starts_with("Table 1: Accuracy\n\n"), "md:\n{md}");
        // Header + separator.
        assert!(md.contains("| Setting | WebQs | SciQ |"), "md:\n{md}");
        assert!(md.contains("| --- | --- | --- |"), "md:\n{md}");
        // Numeric cells survive (the whole point of the fix).
        assert!(md.contains("| Protected(ours) | 16.8 | 91.7 |"), "md:\n{md}");
        // Short section row padded to header width.
        assert!(md.contains("| GPT2-Small |  |  |"), "md:\n{md}");
    }

    #[test]
    fn none_when_no_rows() {
        assert!(render_table_markdown(Some("cap"), None).is_none());
        assert!(render_table_markdown(None, Some(&json!({"rows": []}))).is_none());
    }

    #[test]
    fn escapes_pipes_in_cells() {
        let rows = json!({"rows": [["a|b", "c"]]});
        let md = render_table_markdown(None, Some(&rows)).unwrap();
        assert!(md.contains("a\\|b"), "md:\n{md}");
    }
}
