//! Figure media-fetch endpoint.
//!
//! Streams a captured figure back to clients as WEBP so the UI and agent read
//! paths can render / inline it. Figure bytes are persisted by the VLM-OCR
//! pipeline into the `chunks` table (`kind = 'figure'`, `figure_id` set,
//! `media_bytes` non-null) — see `migrations/052_add_media_to_chunks.sql` and
//! the backfill in `processor/pdf_processing.rs::backfill_figure_media`.
//!
//! Figures captured after the WEBP storage switch are already WEBP and stream
//! unchanged; legacy figures stored as PNG are transcoded to WEBP on read so
//! every consumer gets the same compact format regardless of capture vintage.

use axum::extract::{Path, State};

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

/// Lossy WEBP quality used when transcoding legacy non-WEBP figures on read.
/// Matches the storage-time quality in `edgequake-pdf`'s `figure_extract`.
#[cfg(feature = "postgres")]
const WEBP_QUALITY: u8 = 80;

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
        (status = 200, description = "Figure bytes as WEBP", content_type = "image/webp"),
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

        let stored_mime = mime.unwrap_or_else(|| "image/png".to_string());

        // Always serve WEBP. Figures captured after the storage switch are
        // already WEBP and stream unchanged; legacy PNG figures are transcoded
        // on read. A transcode failure falls back to the stored bytes (with
        // their real MIME) rather than 500-ing — the figure is still useful.
        let (bytes, content_type) = if stored_mime == "image/webp" {
            (bytes, "image/webp".to_string())
        } else {
            match transcode_to_webp(&bytes, WEBP_QUALITY) {
                Some(webp_bytes) => (webp_bytes, "image/webp".to_string()),
                None => (bytes, stored_mime),
            }
        };

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

/// Decode stored figure bytes (PNG/JPEG/…) and re-encode as lossy WEBP at
/// `quality`.
///
/// Returns `None` if the bytes can't be decoded or the re-encode produces
/// nothing — the caller then falls back to streaming the original bytes, so a
/// transcode hiccup never fails the request.
#[cfg(feature = "postgres")]
fn transcode_to_webp(bytes: &[u8], quality: u8) -> Option<Vec<u8>> {
    let img = match image::load_from_memory(bytes) {
        Ok(img) => img,
        Err(e) => {
            tracing::warn!(error = %e, "figure transcode: decode failed; serving stored bytes");
            return None;
        }
    };

    // `webp::Encoder::from_rgb` wants a tightly-packed RGB8 buffer.
    let rgb = img.to_rgb8();
    let encoder = webp::Encoder::from_rgb(rgb.as_raw(), rgb.width(), rgb.height());
    let encoded = encoder.encode(quality.min(100) as f32);
    if encoded.is_empty() {
        tracing::warn!("figure transcode: webp encode produced empty output");
        return None;
    }
    Some(encoded.to_vec())
}

#[cfg(all(test, feature = "postgres"))]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Encode a solid-colour RGB image as PNG for use as transcode input.
    fn png_fixture(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([12, 34, 56]));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn transcodes_png_to_decodable_webp() {
        let png = png_fixture(120, 90);
        let webp = transcode_to_webp(&png, WEBP_QUALITY).expect("png should transcode to webp");
        let decoded = image::load_from_memory(&webp).expect("output should be valid webp");
        assert_eq!((decoded.width(), decoded.height()), (120, 90));
        // The lossy WEBP of a solid colour is far smaller than the PNG input.
        assert!(webp.len() < png.len(), "webp {} should be < png {}", webp.len(), png.len());
    }

    #[test]
    fn returns_none_on_undecodable_bytes() {
        assert!(transcode_to_webp(b"not an image", WEBP_QUALITY).is_none());
    }
}
