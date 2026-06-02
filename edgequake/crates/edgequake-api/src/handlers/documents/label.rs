//! Set / clear a document's user-assigned label.
//!
//! `PUT /api/v1/documents/{document_id}/label` — body `{"label": "Nexus"}`
//! to set, `{"label": null}` (or `{}`) to clear. Mirrors the value into
//! both `documents.label` (postgres, source of truth) and the document's
//! `{id}-metadata` KV record so the list endpoint can surface it without
//! an extra DB roundtrip per row.

use axum::extract::{Path, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::info;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

const LABEL_MAX_LEN: usize = 80;

/// Request body for `PUT /api/v1/documents/{id}/label`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct SetDocumentLabelRequest {
    /// New label. `null` (or absent) clears the label. Length 1..=80 chars
    /// after trimming; whitespace-only is treated as a clear.
    #[serde(default)]
    pub label: Option<String>,
}

/// Response for `PUT /api/v1/documents/{id}/label`.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct SetDocumentLabelResponse {
    pub document_id: String,
    /// Echo back the canonical label after normalisation (or `null` if
    /// cleared). Lets the client update its UI state without reloading.
    pub label: Option<String>,
}

/// `PUT /api/v1/documents/{document_id}/label`
#[utoipa::path(
    put,
    path = "/api/v1/documents/{document_id}/label",
    params(("document_id" = String, Path, description = "Document identifier")),
    request_body = SetDocumentLabelRequest,
    responses(
        (status = 200, description = "Label updated", body = SetDocumentLabelResponse),
        (status = 400, description = "Label too long"),
        (status = 404, description = "Document not found")
    ),
    tag = "Documents"
)]
pub async fn set_document_label(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
    Json(req): Json<SetDocumentLabelRequest>,
) -> ApiResult<Json<SetDocumentLabelResponse>> {
    // Normalise: trim whitespace, treat empty as "clear".
    let normalised: Option<String> = req
        .label
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    if let Some(ref s) = normalised {
        if s.chars().count() > LABEL_MAX_LEN {
            return Err(ApiError::BadRequest(format!(
                "label must be at most {LABEL_MAX_LEN} characters"
            )));
        }
    }

    // Load the document's metadata (also gates 404 + tenant access).
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

    if let Some(ref filter_tid) = tenant_ctx.tenant_id {
        if let Some(doc_tid) = obj.get("tenant_id").and_then(|v| v.as_str()) {
            if doc_tid != filter_tid {
                return Err(ApiError::Forbidden);
            }
        }
    }
    if let Some(ref filter_ws) = tenant_ctx.workspace_id {
        if let Some(doc_ws) = obj.get("workspace_id").and_then(|v| v.as_str()) {
            if doc_ws != filter_ws {
                return Err(ApiError::Forbidden);
            }
        }
    }

    // Persist to postgres `documents.label` when the row exists.
    #[cfg(feature = "postgres")]
    {
        if let Some(ref pdf_storage) = state.pdf_storage {
            if let Ok(doc_uuid) = Uuid::parse_str(&document_id) {
                if let Err(e) = pdf_storage
                    .set_document_label(&doc_uuid, normalised.as_deref())
                    .await
                {
                    tracing::warn!(
                        document_id = %document_id,
                        error = %e,
                        "Failed to update documents.label (KV still updated)"
                    );
                }
            }
        }
    }

    // Mirror into KV metadata so the list endpoint sees the change without
    // a DB roundtrip per row.
    let mut updated = metadata.clone();
    if let Some(obj) = updated.as_object_mut() {
        match &normalised {
            Some(s) => {
                obj.insert(
                    "label".to_string(),
                    serde_json::Value::String(s.clone()),
                );
            }
            None => {
                obj.remove("label");
            }
        }
    }
    state
        .kv_storage
        .upsert(&[(metadata_key, updated)])
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to persist label to KV: {e}")))?;

    info!(
        document_id = %document_id,
        has_label = normalised.is_some(),
        "Document label updated"
    );

    Ok(Json(SetDocumentLabelResponse {
        document_id,
        label: normalised,
    }))
}
