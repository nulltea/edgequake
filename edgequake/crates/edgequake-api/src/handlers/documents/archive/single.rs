//! Single-document archive and unarchive handlers.
//!
//! Archive keeps the curated assets (document row, PDF, Markdown, algorithms,
//! `document_repos`) and drops the rebuildable ones (chunks, embeddings, KG
//! contributions, `reference_codebase_*`). The KV `-metadata` blob is kept
//! and gets an `archived_at` field so KV-iterating callers (`list_documents`,
//! `collect_workspace_documents`) can filter.

use axum::{extract::State, Json};
#[cfg(feature = "postgres")]
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::handlers::documents_types::*;
use crate::state::AppState;

use super::super::storage_helpers::{
    get_workspace_vector_storage_for_delete, resolve_kv_key_prefix, strip_document_from_graph,
};

/// Archive a document by ID.
#[utoipa::path(
    post,
    path = "/api/v1/documents/{document_id}/archive",
    tag = "Documents",
    params(
        ("document_id" = String, Path, description = "Document ID to archive")
    ),
    responses(
        (status = 200, description = "Document archived", body = ArchiveDocumentResponse),
        (status = 404, description = "Document not found"),
        (status = 409, description = "Document already archived")
    )
)]
pub async fn archive_document(
    State(state): State<AppState>,
    axum::extract::Path(document_id): axum::extract::Path<String>,
) -> ApiResult<Json<ArchiveDocumentResponse>> {
    let keys = state.kv_storage.keys().await?;

    let (actual_key_prefix, metadata_key, has_metadata) =
        resolve_kv_key_prefix(&document_id, &keys, &state).await;
    let key_id_mismatch = actual_key_prefix != document_id;

    if !has_metadata {
        return Err(ApiError::NotFound(format!(
            "Document {} not found",
            document_id
        )));
    }

    let metadata = state
        .kv_storage
        .get_by_id(&metadata_key)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("Document {} not found", document_id)))?;

    let metadata_obj = metadata.as_object().ok_or_else(|| {
        ApiError::Internal("Document metadata is not a JSON object".to_string())
    })?;

    if let Some(existing) = metadata_obj.get("archived_at").and_then(|v| v.as_str()) {
        return Err(ApiError::Conflict(format!(
            "Document {} is already archived (at {})",
            document_id, existing
        )));
    }

    let workspace_id_for_storage = metadata_obj
        .get("workspace_id")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();
    let document_status = metadata_obj
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let track_id_opt = metadata_obj
        .get("track_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Cancel any in-flight task so the processor stops writing chunks/entities
    // before we wipe the derived data.
    if matches!(document_status.as_str(), "pending" | "processing") {
        match &track_id_opt {
            Some(track_id) => {
                let cancelled = state.cancellation_registry.cancel(track_id).await;
                tracing::info!(
                    document_id = %document_id,
                    track_id = %track_id,
                    status = %document_status,
                    cancelled,
                    "Cancelled in-flight task before archive"
                );
            }
            None => {
                tracing::warn!(
                    document_id = %document_id,
                    status = %document_status,
                    "No track_id in metadata — proceeding with archive without task cancellation"
                );
            }
        }
    }

    let chunk_prefix = format!("{}-chunk-", actual_key_prefix);
    let chunk_ids: Vec<String> = keys
        .iter()
        .filter(|k| k.starts_with(&chunk_prefix))
        .cloned()
        .collect();
    let chunks_deleted = chunk_ids.len();

    let workspace_vector_storage =
        get_workspace_vector_storage_for_delete(&state, &workspace_id_for_storage).await;

    let mut embeddings_deleted = 0usize;
    if !chunk_ids.is_empty() {
        if let Err(e) = workspace_vector_storage.delete(&chunk_ids).await {
            tracing::warn!(
                document_id = %document_id,
                error = %e,
                "Failed to delete chunk embeddings during archive; continuing"
            );
        } else {
            embeddings_deleted = chunk_ids.len();
        }
    }

    // Strip this document from every KG entity/edge.
    let source_prefixes: Vec<String> = if key_id_mismatch {
        vec![actual_key_prefix.clone(), document_id.clone()]
    } else {
        vec![document_id.clone()]
    };
    let graph_stats =
        strip_document_from_graph(&state, &source_prefixes, &workspace_vector_storage).await?;

    // KV cleanup: drop chunks, content, and lineage; keep -metadata.
    let mut keys_to_delete: Vec<String> = chunk_ids.clone();
    let content_key = format!("{}-content", actual_key_prefix);
    if keys.contains(&content_key) {
        keys_to_delete.push(content_key);
    }
    let lineage_key = format!("{}-lineage", actual_key_prefix);
    if keys.contains(&lineage_key) {
        keys_to_delete.push(lineage_key);
    }
    let extra_prefix_keys: Vec<String> = keys
        .iter()
        .filter(|k| {
            k.starts_with(&format!("{}-", actual_key_prefix))
                && !keys_to_delete.contains(k)
                && **k != metadata_key
        })
        .cloned()
        .collect();
    keys_to_delete.extend(extra_prefix_keys);

    if key_id_mismatch {
        let alt_prefix_keys: Vec<String> = keys
            .iter()
            .filter(|k| {
                k.starts_with(&format!("{}-", document_id))
                    && !keys_to_delete.contains(k)
                    && **k != metadata_key
            })
            .cloned()
            .collect();
        keys_to_delete.extend(alt_prefix_keys);
    }

    if !keys_to_delete.is_empty() {
        state.kv_storage.delete(&keys_to_delete).await?;
    }

    // Postgres: chunks, reference_codebase_*, mark documents.archived_at.
    let mut indexed_code_deleted: u64 = 0;
    #[cfg(feature = "postgres")]
    {
        if let Some(ref pdf_storage) = state.pdf_storage {
            let doc_ids_to_try: Vec<&str> = if key_id_mismatch {
                vec![&actual_key_prefix, &document_id]
            } else {
                vec![&document_id]
            };
            let mut archived_pg = false;
            for doc_id_str in &doc_ids_to_try {
                if let Ok(doc_uuid) = Uuid::parse_str(doc_id_str) {
                    let _ = pdf_storage.delete_chunks_for_document(&doc_uuid).await;
                    match pdf_storage.archive_document(&doc_uuid).await {
                        Ok(_) => {
                            archived_pg = true;
                            break;
                        }
                        Err(e) => {
                            tracing::warn!(
                                document_id = %doc_id_str,
                                error = %e,
                                "Failed to mark documents.archived_at; trying next ID"
                            );
                        }
                    }
                }
            }
            if !archived_pg {
                tracing::warn!(
                    document_id = %document_id,
                    "No matching documents row found to archive — KV-only document"
                );
            }

            for doc_id_str in &doc_ids_to_try {
                match pdf_storage
                    .delete_reference_codebase_for_document(doc_id_str)
                    .await
                {
                    Ok(n) => indexed_code_deleted += n,
                    Err(e) => {
                        tracing::warn!(
                            document_id = %doc_id_str,
                            error = %e,
                            "Failed to delete reference_codebase rows; continuing"
                        );
                    }
                }
            }
        }
    }

    // Write the archived flag into the KV metadata so KV-iterating callers
    // (list_documents, collect_workspace_documents) can filter cheaply.
    let archived_at = chrono::Utc::now().to_rfc3339();
    let mut updated_metadata = metadata.clone();
    if let Some(obj) = updated_metadata.as_object_mut() {
        obj.insert(
            "archived_at".to_string(),
            serde_json::Value::String(archived_at.clone()),
        );
        obj.insert("chunk_count".to_string(), serde_json::json!(0));
        obj.insert("entity_count".to_string(), serde_json::json!(0));
        obj.insert("relationship_count".to_string(), serde_json::json!(0));
    }
    state
        .kv_storage
        .upsert(&[(metadata_key.clone(), updated_metadata)])
        .await?;

    tracing::info!(
        document_id = %document_id,
        chunks_deleted,
        embeddings_deleted,
        entities_removed = graph_stats.entities_removed,
        entities_updated = graph_stats.entities_updated,
        relationships_removed = graph_stats.relationships_removed,
        relationships_updated = graph_stats.relationships_updated,
        indexed_code_deleted,
        "Document archived"
    );

    Ok(Json(ArchiveDocumentResponse {
        document_id,
        archived: true,
        archived_at: Some(archived_at),
        chunks_deleted,
        embeddings_deleted,
        entities_affected: graph_stats.entities_removed + graph_stats.entities_updated,
        relationships_affected: graph_stats.relationships_removed
            + graph_stats.relationships_updated,
        indexed_code_deleted,
    }))
}

/// Unarchive a document by ID.
///
/// Clears `archived_at` in both the documents row and the KV metadata. The
/// document's derived data (chunks/embeddings/KG/indexed code) is not
/// regenerated here — the caller is expected to trigger a workspace-level
/// rebuild to repopulate.
#[utoipa::path(
    post,
    path = "/api/v1/documents/{document_id}/unarchive",
    tag = "Documents",
    params(
        ("document_id" = String, Path, description = "Document ID to unarchive")
    ),
    responses(
        (status = 200, description = "Document unarchived", body = UnarchiveDocumentResponse),
        (status = 404, description = "Document not found")
    )
)]
pub async fn unarchive_document(
    State(state): State<AppState>,
    axum::extract::Path(document_id): axum::extract::Path<String>,
) -> ApiResult<Json<UnarchiveDocumentResponse>> {
    let keys = state.kv_storage.keys().await?;
    let (actual_key_prefix, metadata_key, has_metadata) =
        resolve_kv_key_prefix(&document_id, &keys, &state).await;
    let key_id_mismatch = actual_key_prefix != document_id;

    if !has_metadata {
        return Err(ApiError::NotFound(format!(
            "Document {} not found",
            document_id
        )));
    }

    let metadata = state
        .kv_storage
        .get_by_id(&metadata_key)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("Document {} not found", document_id)))?;

    let mut updated_metadata = metadata.clone();
    if let Some(obj) = updated_metadata.as_object_mut() {
        obj.remove("archived_at");
    }
    state
        .kv_storage
        .upsert(&[(metadata_key, updated_metadata)])
        .await?;

    #[cfg(feature = "postgres")]
    {
        if let Some(ref pdf_storage) = state.pdf_storage {
            let doc_ids_to_try: Vec<&str> = if key_id_mismatch {
                vec![&actual_key_prefix, &document_id]
            } else {
                vec![&document_id]
            };
            for doc_id_str in &doc_ids_to_try {
                if let Ok(doc_uuid) = Uuid::parse_str(doc_id_str) {
                    if pdf_storage.unarchive_document(&doc_uuid).await.is_ok() {
                        break;
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "postgres"))]
    let _ = key_id_mismatch;

    tracing::info!(document_id = %document_id, "Document unarchived");

    Ok(Json(UnarchiveDocumentResponse {
        document_id,
        unarchived: true,
    }))
}
