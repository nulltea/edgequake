//! List a document's parsed references (citations).
//!
//! `GET /api/v1/documents/{document_id}/references` returns the references
//! parsed from the document's reference section during ingestion, ordered by
//! reference number. Read-only — references are deterministic parser output
//! (no review/approval workflow). See `edgequake_pipeline::references`.

use axum::extract::{Path, State};
use axum::Json;
use serde::Serialize;
use utoipa::ToSchema;

use crate::error::ApiResult;
use crate::middleware::TenantContext;
use crate::state::AppState;

/// One reference in the API response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct DocumentReferenceDto {
    pub reference_number: i32,
    pub raw_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doi: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Response for `GET /api/v1/documents/{document_id}/references`.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct DocumentReferencesResponse {
    pub document_id: String,
    pub references: Vec<DocumentReferenceDto>,
    pub total: usize,
}

/// `GET /api/v1/documents/{document_id}/references`
#[utoipa::path(
    get,
    path = "/api/v1/documents/{document_id}/references",
    params(("document_id" = String, Path, description = "Document identifier")),
    responses(
        (status = 200, description = "Parsed references", body = DocumentReferencesResponse)
    ),
    tag = "Documents"
)]
pub async fn list_document_references(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
) -> ApiResult<Json<DocumentReferencesResponse>> {
    list_document_references_impl(state, tenant_ctx, document_id).await
}

#[cfg(feature = "postgres")]
async fn list_document_references_impl(
    state: AppState,
    tenant_ctx: TenantContext,
    document_id: String,
) -> ApiResult<Json<DocumentReferencesResponse>> {
    use crate::error::ApiError;
    use edgequake_storage::traits::ReferenceStorage;
    use edgequake_storage::PgReferenceStorage;
    use uuid::Uuid;

    let tenant_id = tenant_ctx
        .tenant_id
        .as_ref()
        .ok_or_else(|| ApiError::BadRequest("tenant_id is required".to_string()))
        .and_then(|s| {
            Uuid::parse_str(s).map_err(|e| ApiError::BadRequest(format!("Invalid tenant_id: {e}")))
        })?;
    let workspace_id = tenant_ctx
        .workspace_id
        .as_ref()
        .ok_or_else(|| ApiError::BadRequest("workspace_id is required".to_string()))
        .and_then(|s| {
            Uuid::parse_str(s)
                .map_err(|e| ApiError::BadRequest(format!("Invalid workspace_id: {e}")))
        })?;

    let pool = state.pg_pool.as_ref().ok_or_else(|| {
        ApiError::Internal("Reference storage requires PostgreSQL pool".to_string())
    })?;
    let storage = PgReferenceStorage::new(pool.clone());

    let rows = storage
        .list_references(tenant_id, workspace_id, &document_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list references: {e}")))?;

    let references: Vec<DocumentReferenceDto> = rows
        .into_iter()
        .map(|r| DocumentReferenceDto {
            reference_number: r.reference_number,
            raw_text: r.raw_text,
            doi: r.doi,
            url: r.url,
        })
        .collect();
    let total = references.len();

    Ok(Json(DocumentReferencesResponse {
        document_id,
        references,
        total,
    }))
}

#[cfg(not(feature = "postgres"))]
async fn list_document_references_impl(
    _state: AppState,
    _tenant_ctx: TenantContext,
    document_id: String,
) -> ApiResult<Json<DocumentReferencesResponse>> {
    Ok(Json(DocumentReferencesResponse {
        document_id,
        references: Vec::new(),
        total: 0,
    }))
}
