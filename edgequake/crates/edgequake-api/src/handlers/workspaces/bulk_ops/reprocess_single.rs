//! Per-document reprocess endpoint.
//!
//! `POST /api/v1/documents/{document_id}/reprocess` re-runs a single document
//! through the ingestion pipeline. With `skip_extraction = true` (the default
//! for this endpoint), it re-parses, re-chunks, and re-embeds **without** the
//! heavy LLM stages (entity/relationship/algorithm/repo/table extraction) —
//! making the document queryable on chunk embeddings quickly and cheaply. Any
//! pre-existing entities/relationships are left untouched.
//!
//! Lives under `bulk_ops` to reuse the shared task builders
//! ([`super::build_reprocess_task`], [`super::mark_document_pending`],
//! [`super::DocumentInfo`]); the route itself is registered under `/documents`.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

use super::{build_reprocess_task, mark_document_pending, DocumentInfo};

/// Request body for the per-document reprocess endpoint. The body is optional;
/// when omitted, `skip_extraction` defaults to `true` (chunks-only reprocess).
#[derive(Debug, Clone, Deserialize, utoipa::ToSchema)]
pub struct ReprocessDocumentRequest {
    /// When true (default), guard out the heavy LLM stages — re-embed chunks
    /// only. When false, run the full pipeline.
    #[serde(default = "default_skip_extraction")]
    pub skip_extraction: bool,
}

fn default_skip_extraction() -> bool {
    true
}

impl Default for ReprocessDocumentRequest {
    fn default() -> Self {
        Self {
            skip_extraction: default_skip_extraction(),
        }
    }
}

/// Response for the per-document reprocess endpoint.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct ReprocessDocumentResponse {
    pub document_id: String,
    pub status: String,
    pub track_id: String,
    pub skip_extraction: bool,
}

/// `POST /api/v1/documents/{document_id}/reprocess`
///
/// Re-queue a single document for processing. Defaults to chunks-only
/// (`skip_extraction = true`).
#[utoipa::path(
    post,
    path = "/api/v1/documents/{document_id}/reprocess",
    params(("document_id" = String, Path, description = "Document identifier")),
    request_body = Option<ReprocessDocumentRequest>,
    responses(
        (status = 202, description = "Document re-queued for processing", body = ReprocessDocumentResponse),
        (status = 400, description = "Could not resolve workspace/tenant or build a task"),
        (status = 404, description = "Document not found")
    ),
    tag = "Documents"
)]
pub async fn reprocess_document(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
    // Optional body: a missing/empty body must not 400 (mirrors `reprocess_failed`).
    body: Option<Json<ReprocessDocumentRequest>>,
) -> ApiResult<(StatusCode, Json<ReprocessDocumentResponse>)> {
    use chrono::Utc;

    let request = body.map(|b| b.0).unwrap_or_default();

    // Load the document's metadata object.
    let metadata_key = format!("{}-metadata", document_id);
    let metadata = state
        .kv_storage
        .get_by_id(&metadata_key)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to fetch document metadata: {e}")))?
        .ok_or_else(|| ApiError::NotFound(format!("Document not found: {document_id}")))?;
    let obj = metadata
        .as_object()
        .ok_or_else(|| ApiError::Internal("Malformed document metadata".to_string()))?;

    // Resolve workspace/tenant: prefer the document's stored ids, fall back to
    // the caller's context (same precedence as `trigger_extraction`).
    let workspace_id = obj
        .get("workspace_id")
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
        .or_else(|| tenant_ctx.workspace_id_uuid())
        .ok_or_else(|| ApiError::BadRequest("Could not resolve workspace_id".to_string()))?;

    let workspace = state
        .workspace_service
        .get_workspace(workspace_id)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?
        .ok_or_else(|| ApiError::NotFound(format!("Workspace {workspace_id} not found")))?;

    // BR0201: verify workspace belongs to the requesting tenant.
    if let Some(ref ctx_tid) = tenant_ctx.tenant_id {
        if workspace.tenant_id.to_string() != *ctx_tid {
            return Err(ApiError::NotFound(format!(
                "Document not found: {document_id}"
            )));
        }
    }

    // Build a single DocumentInfo from the loaded metadata.
    let doc = DocumentInfo {
        doc_id: document_id.clone(),
        title: obj
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or(&document_id)
            .to_string(),
        chunk_count: obj.get("chunk_count").and_then(|v| v.as_u64()).unwrap_or(1) as usize,
        source_type: obj
            .get("source_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        pdf_id_str: obj
            .get("pdf_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        status: obj
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        extraction_skipped: obj
            .get("extraction_skipped")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    };

    let track_id = format!(
        "reprocess_{}_{}",
        Utc::now().format("%Y%m%d_%H%M%S"),
        &Uuid::new_v4().to_string()[..8]
    );

    mark_document_pending(&state, &doc.doc_id, &track_id).await;

    let extra_meta = serde_json::Map::new();
    let (task_type, task_value) = build_reprocess_task(
        &state,
        &workspace,
        workspace_id,
        &doc,
        &track_id,
        extra_meta,
        request.skip_extraction,
    )
    .await
    .ok_or_else(|| {
        ApiError::BadRequest(format!(
            "No reprocessable content for document {document_id} (no PDF bytes or stored text)"
        ))
    })?;

    let task = edgequake_tasks::Task::new(workspace.tenant_id, workspace_id, task_type, task_value);
    state
        .task_storage
        .create_task(&task)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to create reprocess task: {e}")))?;
    state
        .task_queue
        .send(task)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to queue reprocess task: {e}")))?;

    Ok((
        StatusCode::ACCEPTED,
        Json(ReprocessDocumentResponse {
            document_id,
            status: "processing".to_string(),
            track_id,
            skip_extraction: request.skip_extraction,
        }),
    ))
}
