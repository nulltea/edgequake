//! POST /api/v1/graph/entities/{entity_name}/body
//!
//! Plan-4 endpoint: attach a body-text blob to an existing graph entity.
//!
//! Edgequake's entity model originally had no concept of a body — entity
//! nodes carried only `entity_type`, `description`, etc., while document
//! body text lived in the chunks table. Lattice (and other clients that
//! bypass the doc-ingestion pipeline) need somewhere to store the actual
//! markdown / code / quoted text that belongs to a node so the body is
//! recoverable from the graph alone.
//!
//! Body is stored as a property on the AGE node (`body` key). Idempotent
//! overwrite. Optionally regenerates the entity's embedding vector using
//! the workspace's configured embedding provider so the body text becomes
//! searchable.

use axum::{
    extract::{Path, State},
    Json,
};
use chrono::Utc;

use crate::error::{ApiError, ApiResult};
use crate::handlers::entities_types::{SetEntityBodyRequest, SetEntityBodyResponse};
use crate::middleware::TenantContext;
use crate::state::AppState;

use super::embed::{default_embedding_text_with_body, embed_entity_and_store};
use super::normalize_entity_name;

/// Attach (or replace) the body text on an existing entity.
#[utoipa::path(
    post,
    path = "/api/v1/graph/entities/{entity_name}/body",
    tag = "Entities",
    params(
        ("entity_name" = String, Path, description = "Entity name (normalized server-side)")
    ),
    request_body = SetEntityBodyRequest,
    responses(
        (status = 200, description = "Body attached", body = SetEntityBodyResponse),
        (status = 404, description = "Entity not found"),
    ),
)]
pub async fn set_entity_body(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(entity_name): Path<String>,
    Json(req): Json<SetEntityBodyRequest>,
) -> ApiResult<Json<SetEntityBodyResponse>> {
    let entity_name = normalize_entity_name(&entity_name);

    // Load entity so we can attach the body to its existing properties.
    let mut node = state
        .graph_storage
        .get_node(&entity_name)
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("Entity '{}' not found", entity_name)))?;

    let body_length = req.body.len();
    node.properties
        .insert("body".to_string(), req.body.clone().into());

    // Track update time.
    let now = Utc::now().to_rfc3339();
    node.properties.insert("updated_at".to_string(), now.into());

    state
        .graph_storage
        .upsert_node(&entity_name, node.properties.clone())
        .await?;

    let mut embedded = false;
    if req.embed.unwrap_or(false) {
        let description = node
            .properties
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let entity_type = node
            .properties
            .get("entity_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let source_id = node
            .properties
            .get("source_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let text = req.embedding_text.clone().unwrap_or_else(|| {
            default_embedding_text_with_body(&entity_name, &description, &req.body)
        });
        embedded = embed_entity_and_store(
            &state,
            &tenant_ctx,
            &entity_name,
            &entity_type,
            &description,
            &source_id,
            &text,
        )
        .await?;
    }

    Ok(Json(SetEntityBodyResponse {
        status: "success".to_string(),
        message: "Body attached".to_string(),
        entity_name,
        body_length,
        embedded,
    }))
}
