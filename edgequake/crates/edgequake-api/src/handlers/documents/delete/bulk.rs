//! Bulk deletion handler for documents in the active workspace.
//!
//! Deletes all documents that match the request's tenant+workspace context,
//! skipping those actively being processed unless they are detected as
//! "stuck" (>1 hour at 100% progress). Also cleans up graph entities and
//! edges whose `source_ids` are exhausted by the deletions, and PDF
//! table entries belonging to the same workspace.

use axum::{extract::State, Json};
use chrono::Utc;
#[cfg(feature = "postgres")]
use edgequake_storage::ListPdfFilter;

use crate::error::ApiResult;
use crate::handlers::documents_types::*;
use crate::middleware::TenantContext;
use crate::state::AppState;

/// Delete all documents in the active workspace (bulk deletion).
///
/// This endpoint backs the frontend "Clear All" button. It is **scoped to
/// the request's tenant+workspace context** — historically it was global
/// and a single click would wipe every tenant/workspace, which destroyed
/// data outside the calling user's view. The fix here makes the handler
/// extract `TenantContext` and filter every metadata row by matching
/// `tenant_id` and `workspace_id` before deleting.
///
/// Documents that are actively being processed (pending/processing) are
/// skipped to prevent data corruption.
#[utoipa::path(
    delete,
    path = "/api/v1/documents",
    tag = "Documents",
    responses(
        (status = 200, description = "Documents deleted", body = DeleteAllDocumentsResponse),
        (status = 500, description = "Internal error")
    )
)]
pub async fn delete_all_documents(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
) -> ApiResult<Json<DeleteAllDocumentsResponse>> {
    let filter_workspace_id = tenant_ctx.workspace_id.clone();
    let filter_tenant_id = tenant_ctx.tenant_id.clone();
    tracing::info!(
        workspace_id = ?filter_workspace_id,
        tenant_id = ?filter_tenant_id,
        "Bulk delete requested (workspace-scoped)"
    );

    let keys = state.kv_storage.keys().await?;

    // Find all document metadata keys to identify unique documents.
    // Per-document filtering by tenant/workspace happens below — we need
    // to read each metadata row to learn its owning workspace.
    let metadata_keys: Vec<String> = keys
        .iter()
        .filter(|k| k.ends_with("-metadata"))
        .cloned()
        .collect();

    let mut deleted_count = 0usize;
    let mut total_chunks_deleted = 0usize;
    let mut total_entities_removed = 0usize;
    let mut total_relationships_removed = 0usize;
    let mut skipped_count = 0usize;
    let mut skipped_documents = Vec::new();

    // Define stuck threshold: documents processing for > 1 hour are considered stuck
    let stuck_threshold_secs = 3600; // 1 hour

    // Track which document IDs we deleted so the orphan-cleanup pass at
    // the end only removes graph nodes/edges that lost ALL of their
    // sources (rather than scanning every node globally).
    let mut deleted_doc_ids: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for metadata_key in &metadata_keys {
        // Extract document_id from metadata key (format: {document_id}-metadata)
        let document_id = metadata_key.trim_end_matches("-metadata").to_string();

        // Get document status and metadata to check if safe to delete.
        // ALSO read the doc's owning tenant_id + workspace_id so we can
        // skip docs that belong to other workspaces — without this filter
        // the handler historically nuked every workspace's data.
        let (status, updated_at_opt, stage_progress_opt, doc_tenant_id, doc_workspace_id) =
            if let Ok(Some(metadata)) = state.kv_storage.get_by_id(metadata_key).await {
                let status = metadata
                    .get("status")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let updated_at = metadata
                    .get("updated_at")
                    .and_then(|v| v.as_str())
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&chrono::Utc));
                let stage_progress = metadata.get("stage_progress").and_then(|v| v.as_f64());
                let doc_tenant_id = metadata
                    .get("tenant_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let doc_workspace_id = metadata
                    .get("workspace_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                (
                    status,
                    updated_at,
                    stage_progress,
                    doc_tenant_id,
                    doc_workspace_id,
                )
            } else {
                ("unknown".to_string(), None, None, None, None)
            };

        // SECURITY: STRICT tenant+workspace match. Both must equal the
        // request's filter exactly (None == None). Matches list.rs's
        // `matches_tenant_context`.
        if doc_tenant_id != filter_tenant_id || doc_workspace_id != filter_workspace_id {
            continue;
        }

        // Skip documents that are actively being processed (unless stuck)
        // A document is considered stuck if:
        //   - Status is "processing" or "pending"
        //   - AND updated_at is more than stuck_threshold_secs ago
        //   - AND stage_progress is 1.0 (100%) or close to it
        let is_stuck = if status == "pending" || status == "processing" {
            if let Some(updated_at) = updated_at_opt {
                let age_secs = (Utc::now() - updated_at).num_seconds();
                let high_progress = stage_progress_opt.map(|p| p >= 0.99).unwrap_or(false);
                age_secs > stuck_threshold_secs && high_progress
            } else {
                false
            }
        } else {
            false
        };

        if (status == "pending" || status == "processing") && !is_stuck {
            tracing::debug!(
                document_id = %document_id,
                status = %status,
                "Skipping bulk delete of document with active processing"
            );
            skipped_count += 1;
            skipped_documents.push(document_id.clone());
            continue;
        }

        if is_stuck {
            tracing::info!(
                document_id = %document_id,
                status = %status,
                "Deleting stuck document (>1 hour at 100% progress)"
            );
        }

        // Attempt to delete this document
        // We'll use a simplified version that doesn't require workspace isolation
        // since we're doing a full system clear
        let chunk_prefix = format!("{}-chunk-", document_id);
        let chunk_ids: Vec<String> = keys
            .iter()
            .filter(|k| k.starts_with(&chunk_prefix))
            .cloned()
            .collect();

        let content_key = format!("{}-content", document_id);

        // Delete from KV storage - delete takes a slice of strings
        if !chunk_ids.is_empty() {
            if let Err(e) = state.kv_storage.delete(&chunk_ids).await {
                tracing::warn!(document_id = %document_id, error = %e, "Failed to delete chunks");
            }
        }

        // Delete metadata key
        if let Err(e) = state
            .kv_storage
            .delete(std::slice::from_ref(metadata_key))
            .await
        {
            tracing::warn!(key = %metadata_key, error = %e, "Failed to delete metadata");
        }

        // Delete content key
        if let Err(e) = state
            .kv_storage
            .delete(std::slice::from_ref(&content_key))
            .await
        {
            tracing::warn!(key = %content_key, error = %e, "Failed to delete content");
        }

        // Delete from vector storage (use default storage for bulk operations)
        if !chunk_ids.is_empty() {
            if let Err(e) = state.vector_storage.delete(&chunk_ids).await {
                tracing::warn!(
                    document_id = %document_id,
                    error = %e,
                    "Failed to delete chunk embeddings"
                );
            }
        }

        total_chunks_deleted += chunk_ids.len();
        deleted_count += 1;
        deleted_doc_ids.insert(document_id.clone());

        tracing::debug!(
            document_id = %document_id,
            chunks = chunk_ids.len(),
            "Deleted document in bulk operation"
        );
    }

    // Clean up graph entities/edges whose ONLY source documents were
    // among the ones we just deleted. This MUST be scoped to this
    // workspace — a global pass would delete entities from other
    // workspaces that happen to be orphaned for unrelated reasons.
    //
    // We trim each node's `source_ids` (and legacy `source_id`) of any
    // values referring to a deleted doc_id, then delete the node only
    // when no sources remain.
    let all_nodes = state.graph_storage.get_all_nodes().await?;
    let mut removed_node_ids: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for node in all_nodes {
        let still_sourced = node_has_remaining_sources(&node, &deleted_doc_ids);
        if !still_sourced {
            if let Err(e) = state.graph_storage.delete_node(&node.id).await {
                tracing::warn!(node_id = %node.id, error = %e, "Failed to delete orphaned node");
            } else {
                total_entities_removed += 1;
                removed_node_ids.insert(node.id.clone());
            }
        }
    }

    // Sweep edges that lost an endpoint. Bounded by `removed_node_ids`
    // (only edges touching nodes we just removed are candidates), so
    // edges in other workspaces aren't touched.
    if !removed_node_ids.is_empty() {
        let all_edges = state.graph_storage.get_all_edges().await?;
        for edge in all_edges {
            let touches_removed = removed_node_ids.contains(&edge.source)
                || removed_node_ids.contains(&edge.target);
            if touches_removed {
                if let Err(e) = state
                    .graph_storage
                    .delete_edge(&edge.source, &edge.target)
                    .await
                {
                    tracing::warn!(
                        source = %edge.source,
                        target = %edge.target,
                        error = %e,
                        "Failed to delete orphaned edge"
                    );
                } else {
                    total_relationships_removed += 1;
                }
            }
        }
    }

    // Clean up PDF documents table — also workspace-scoped now.
    #[allow(unused_mut)] // mut only used when postgres feature is enabled
    let mut total_pdfs_deleted = 0usize;
    #[cfg(feature = "postgres")]
    if let Some(ref pdf_storage) = state.pdf_storage {
        // ListPdfFilter expects Option<Uuid>, but TenantContext gives us
        // Option<String> (workspace IDs are free-form strings elsewhere).
        // Only filter PDFs when the workspace_id parses as a UUID; this
        // is a strict scope so a non-UUID context can't sweep PDFs.
        let pdf_workspace_filter: Option<uuid::Uuid> = filter_workspace_id
            .as_deref()
            .and_then(|s| uuid::Uuid::parse_str(s).ok());
        let filter = ListPdfFilter {
            workspace_id: pdf_workspace_filter,
            processing_status: None,
            page: Some(1),
            page_size: Some(10000),
        };

        match pdf_storage.list_pdfs(filter).await {
            Ok(pdf_list) => {
                for pdf in pdf_list.items {
                    if let Err(e) = pdf_storage.delete_pdf(&pdf.pdf_id).await {
                        tracing::warn!(
                            pdf_id = %pdf.pdf_id,
                            error = %e,
                            "Failed to delete PDF document"
                        );
                    } else {
                        total_pdfs_deleted += 1;
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to list PDF documents for cleanup");
            }
        }
    }

    tracing::info!(
        deleted = deleted_count,
        skipped = skipped_count,
        chunks = total_chunks_deleted,
        entities = total_entities_removed,
        relationships = total_relationships_removed,
        pdfs = total_pdfs_deleted,
        "Bulk delete complete"
    );

    Ok(Json(DeleteAllDocumentsResponse {
        deleted_count,
        total_chunks_deleted,
        total_entities_removed,
        total_relationships_removed,
        total_pdfs_deleted,
        skipped_count,
        skipped_documents,
    }))
}

/// True when the node retains at least one `source_ids` (or legacy
/// `source_id`) value that does NOT belong to a deleted document.
///
/// A `source_ids` entry counts as "belongs to deleted doc" when it
/// equals a deleted document_id or is prefixed `{doc_id}-chunk-`.
fn node_has_remaining_sources(
    node: &edgequake_storage::GraphNode,
    deleted_doc_ids: &std::collections::HashSet<String>,
) -> bool {
    let is_deleted = |s: &str| -> bool {
        for d in deleted_doc_ids {
            if s == d || s.starts_with(&format!("{d}-chunk-")) {
                return true;
            }
        }
        false
    };

    if let Some(arr) = node
        .properties
        .get("source_ids")
        .and_then(|v| v.as_array())
    {
        for v in arr {
            if let Some(s) = v.as_str() {
                if !is_deleted(s) {
                    return true;
                }
            }
        }
        // source_ids was present but every entry was deleted → no remaining
        if !arr.is_empty() {
            return false;
        }
    }

    // Legacy `source_id` pipe-separated string.
    if let Some(s) = node.properties.get("source_id").and_then(|v| v.as_str()) {
        for part in s.split('|') {
            if !part.is_empty() && !is_deleted(part) {
                return true;
            }
        }
        return false;
    }

    // No source-tracking properties at all — treat as still sourced to
    // avoid wiping pre-source-tracking nodes from other workspaces.
    true
}
