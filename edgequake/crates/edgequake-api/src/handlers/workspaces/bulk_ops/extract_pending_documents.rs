//! Extract-all-pending endpoint.
//!
//! Triggers the heavy LLM extraction stages for every document in a workspace
//! that was ingested in chunks-only mode (`extraction_skipped == true`).
//! Reuses the per-document orchestration in `documents::extraction`.

use axum::{
    extract::{Path, State},
    Json,
};
use serde::Serialize;
use uuid::Uuid;

use crate::error::ApiError;
use crate::handlers::documents::extraction::queue_extraction_tasks;
use crate::middleware::TenantContext;
use crate::state::AppState;

use super::collect_workspace_documents;

/// Response for the workspace-wide extract-pending endpoint.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct ExtractPendingResponse {
    pub workspace_id: Uuid,
    pub status: String,
    /// Documents found that were awaiting extraction.
    pub documents_found: usize,
    /// Documents for which at least one extraction stage was queued.
    pub documents_queued: usize,
    /// Total heavy-stage tasks queued across all documents.
    pub stages_queued: usize,
}

/// Trigger extraction for all chunks-only documents in a workspace.
#[utoipa::path(
    post,
    path = "/api/v1/workspaces/{workspace_id}/extract-pending",
    params(("workspace_id" = Uuid, Path, description = "Workspace ID")),
    responses(
        (status = 200, description = "Extraction queued for pending documents", body = ExtractPendingResponse),
        (status = 404, description = "Workspace not found"),
    ),
    tags = ["workspaces"]
)]
pub async fn extract_pending_documents(
    State(state): State<AppState>,
    Path(workspace_id): Path<Uuid>,
    tenant_ctx: TenantContext,
) -> Result<Json<ExtractPendingResponse>, ApiError> {
    use tracing::info;

    // 1. Verify workspace exists.
    let workspace = state
        .workspace_service
        .get_workspace(workspace_id)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?
        .ok_or_else(|| ApiError::NotFound(format!("Workspace {workspace_id} not found")))?;

    // BR0201: verify workspace belongs to the requesting tenant.
    if let Some(ref ctx_tid) = tenant_ctx.tenant_id {
        if workspace.tenant_id.to_string() != *ctx_tid {
            tracing::warn!(workspace_id = %workspace_id, "Tenant isolation: extract-pending rejected");
            return Err(ApiError::NotFound(format!(
                "Workspace {workspace_id} not found"
            )));
        }
    }

    // 2. Collect workspace documents and keep only those awaiting extraction.
    let docs = collect_workspace_documents(&state, &workspace_id, &workspace.slug).await?;
    let pending: Vec<_> = docs.into_iter().filter(|d| d.extraction_skipped).collect();
    let documents_found = pending.len();

    info!(
        workspace_id = %workspace_id,
        documents_found,
        "extract-pending: triggering extraction for chunks-only documents"
    );

    // 3. Queue extraction for each pending document.
    let mut documents_queued = 0usize;
    let mut stages_queued = 0usize;
    for doc in &pending {
        let metadata_key = format!("{}-metadata", doc.doc_id);
        let Some(metadata) = state
            .kv_storage
            .get_by_id(&metadata_key)
            .await
            .ok()
            .flatten()
        else {
            continue;
        };
        let Some(obj) = metadata.as_object() else {
            continue;
        };
        let queued =
            queue_extraction_tasks(&state, workspace.tenant_id, workspace_id, &doc.doc_id, obj)
                .await;
        if !queued.is_empty() {
            documents_queued += 1;
            stages_queued += queued.len();
        }
    }

    Ok(Json(ExtractPendingResponse {
        workspace_id,
        status: if documents_queued > 0 {
            "processing".to_string()
        } else {
            "no_documents".to_string()
        },
        documents_found,
        documents_queued,
        stages_queued,
    }))
}
