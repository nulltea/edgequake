//! Plan 4: server-side embedding generation for entities created via the
//! direct graph API (POST/PUT /graph/entities, POST /graph/entities/{id}/body).
//!
//! Mirrors the entity-embedding storage that today's document ingestion
//! pipeline performs (see `handlers/documents/upload/text_upload.rs`), so
//! entities created by external graph-writing clients (lattice, etc.) become
//! discoverable via the workspace's vector storage and `POST /query`.
//!
//! Without this path, `POST /graph/entities` would only land a node in the
//! AGE graph — no vector — and the entity would be invisible to semantic
//! search.

use std::sync::Arc;
use tracing::{debug, warn};

use crate::error::{ApiError, ApiResult};
use crate::handlers::query::{get_workspace_embedding_provider, get_workspace_vector_storage};
use crate::handlers::workspaces::invalidate_workspace_stats_cache;
use crate::middleware::TenantContext;
use crate::state::AppState;
use uuid::Uuid;

/// Compute the vector storage key for an entity. Matches the format used by
/// `text_insert.rs` so that vectors written via this path are indistinguishable
/// from pipeline-generated ones at query time.
pub(super) fn entity_vector_id(entity_name: &str) -> String {
    format!("entity:{}", entity_name)
}

/// Embed `text` using the workspace's configured embedding provider and
/// upsert the result into the workspace's vector storage. Returns Ok(true)
/// if a vector was actually stored, Ok(false) if the operation was skipped
/// (e.g. workspace has no embedding configured), or Err on storage failure.
///
/// On success, also invalidates the workspace stats cache so the frontend
/// reflects the new entity count.
pub(super) async fn embed_entity_and_store(
    state: &AppState,
    tenant_ctx: &TenantContext,
    entity_name: &str,
    entity_type: &str,
    description: &str,
    source_id: &str,
    text: &str,
) -> ApiResult<bool> {
    // Workspace context is required: without it we can't pick the right
    // embedding provider or vector store.
    let workspace_id = match tenant_ctx.workspace_id.as_deref() {
        Some(ws) => ws,
        None => {
            warn!(
                entity_name = %entity_name,
                "embed_entity_and_store skipped: no workspace_id in tenant context"
            );
            return Ok(false);
        }
    };

    // Resolve workspace embedding provider. None ⇒ workspace has no override
    // and the caller should rely on document-pipeline default ingestion
    // (which lattice doesn't use). We treat that as "skip" rather than error
    // so older workspaces don't break unrelated callers.
    let provider: Option<Arc<dyn edgequake_query::EmbeddingProvider>> =
        get_workspace_embedding_provider(state, workspace_id).await?;
    let provider = match provider {
        Some(p) => p,
        None => {
            debug!(
                workspace_id = %workspace_id,
                "embed_entity_and_store skipped: workspace has no embedding provider configured"
            );
            return Ok(false);
        }
    };

    // Same skip semantics for vector storage.
    let vector_storage = match get_workspace_vector_storage(state, workspace_id).await? {
        Some(s) => s,
        None => {
            debug!(
                workspace_id = %workspace_id,
                "embed_entity_and_store skipped: workspace has no vector storage configured"
            );
            return Ok(false);
        }
    };

    // Generate the vector.
    let vector = provider.embed_one(text).await.map_err(|e| {
        ApiError::Internal(format!(
            "Failed to embed entity '{}': {}",
            entity_name, e
        ))
    })?;

    // Build metadata matching the shape produced by document ingestion
    // (see text_upload.rs:498-510) so queries treat both sources identically.
    let mut metadata = serde_json::json!({
        "type": "entity",
        "entity_name": entity_name,
        "entity_type": entity_type,
        "description": description,
        "source_id": source_id,
        "workspace_id": workspace_id,
    });
    if let Some(ref tid) = tenant_ctx.tenant_id {
        metadata["tenant_id"] = serde_json::json!(tid);
    }

    let id = entity_vector_id(entity_name);
    vector_storage
        .upsert(&[(id.clone(), vector, metadata)])
        .await
        .map_err(|e| {
            ApiError::Internal(format!(
                "Failed to upsert entity vector '{}': {}",
                id, e
            ))
        })?;

    // Invalidate the stats cache so frontend counts stay fresh. Best-effort:
    // failures here don't fail the request — the entity is already persisted.
    if let Ok(ws_uuid) = Uuid::parse_str(workspace_id) {
        invalidate_workspace_stats_cache(ws_uuid).await;
    }

    Ok(true)
}

/// Compose a default embedding text from entity fields. Used when the client
/// doesn't supply an explicit `embedding_text`. The chosen format mirrors
/// what lattice's pure composers produce at minimum: `name + " " + description`.
pub(super) fn default_embedding_text(name: &str, description: &str) -> String {
    if description.is_empty() {
        name.to_string()
    } else {
        format!("{} {}", name, description)
    }
}

/// Variant for the body endpoint: includes body text in the fallback.
pub(super) fn default_embedding_text_with_body(
    name: &str,
    description: &str,
    body: &str,
) -> String {
    let mut parts: Vec<&str> = Vec::with_capacity(3);
    if !name.is_empty() {
        parts.push(name);
    }
    if !description.is_empty() {
        parts.push(description);
    }
    if !body.is_empty() {
        parts.push(body);
    }
    parts.join(" ")
}
