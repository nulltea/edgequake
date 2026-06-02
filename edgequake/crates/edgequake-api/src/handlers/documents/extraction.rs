//! Trigger-extraction endpoints.
//!
//! A document ingested in chunks-only mode (`skip_extraction`) is queryable
//! via chunk embeddings but has had none of the heavy LLM stages run. These
//! endpoints kick those stages off later, reusing EdgeQuake's existing
//! standalone task types — no PDF re-OCR:
//!
//! - **Entities / relationships**: `TaskType::Insert` over the stored text
//!   (KV `{id}-content` for text docs, `pdf_documents.markdown_content` for
//!   PDFs). Chunk IDs are deterministic (`{doc_id}-chunk-N`) so the re-chunk +
//!   re-embed is idempotent — no duplicate chunks.
//! - **Algorithms**: `TaskType::AlgorithmExtraction` (PDF → layout detection
//!   from the stored PDF bytes; text → stored chunk contents).
//! - **Reference repos**: `TaskType::RepoDetection` (PDF only).
//! - **Table classification**: `TaskType::TableClassification` (PDF only).
//!
//! Two entry points share the same per-document orchestration
//! ([`queue_extraction_tasks`]): a single-document handler ([`trigger_extraction`])
//! and the workspace-wide bulk handler in `workspaces::bulk_ops`.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

/// One queued heavy-stage task.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct QueuedStage {
    pub stage: String,
    pub track_id: String,
}

/// Response for the single-document trigger endpoint.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct TriggerExtractionResponse {
    pub document_id: String,
    pub status: String,
    pub queued: Vec<QueuedStage>,
}

/// Queue the heavy LLM extraction stages for a single document.
///
/// Best-effort per stage: a stage that can't be built (e.g. no stored content)
/// is skipped, logged, and the others still run. Clears the document's
/// `extraction_skipped` flag and flips its status to `processing` up front so
/// it leaves the "needs extraction" set immediately. Returns the stages that
/// were queued.
///
/// `metadata` is the document's already-loaded `{id}-metadata` object.
pub(crate) async fn queue_extraction_tasks(
    state: &AppState,
    tenant_id: Uuid,
    workspace_id: Uuid,
    document_id: &str,
    metadata: &serde_json::Map<String, serde_json::Value>,
) -> Vec<QueuedStage> {
    use edgequake_tasks::{
        AlgorithmExtractionData, RepoDetectionData, Task, TableClassificationData, TaskType,
        TextInsertData,
    };

    let source_type = metadata
        .get("source_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let is_pdf = source_type == "pdf";
    let pdf_id = metadata
        .get("pdf_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let title = metadata
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or(document_id)
        .to_string();

    // Clear the flag + flip status so the document leaves the "needs
    // extraction" set right away (the heavy tasks below run in the
    // background and will write their own terminal status).
    let metadata_key = format!("{}-metadata", document_id);
    {
        let mut updated = metadata.clone();
        updated.insert("extraction_skipped".to_string(), serde_json::json!(false));
        updated.insert("status".to_string(), serde_json::json!("processing"));
        updated.insert("current_stage".to_string(), serde_json::json!("extracting"));
        updated.insert(
            "stage_message".to_string(),
            serde_json::json!("Running extraction..."),
        );
        updated.insert(
            "updated_at".to_string(),
            serde_json::json!(chrono::Utc::now().to_rfc3339()),
        );
        let _ = state
            .kv_storage
            .upsert(&[(metadata_key, serde_json::json!(updated))])
            .await;
    }

    let mut queued: Vec<QueuedStage> = Vec::new();

    let enqueue = |state: &AppState, task: Task| {
        let storage = state.task_storage.clone();
        let queue = state.task_queue.clone();
        async move {
            let track_id = task.track_id.clone();
            storage
                .create_task(&task)
                .await
                .map_err(|e| format!("create_task: {e}"))?;
            queue
                .send(task)
                .await
                .map_err(|e| format!("queue send: {e}"))?;
            Ok::<String, String>(track_id)
        }
    };

    // ── 1. Entity / relationship extraction (Insert over stored text) ──
    let text: Option<String> = if is_pdf {
        load_pdf_markdown(state, document_id, workspace_id).await
    } else {
        let content_key = format!("{}-content", document_id);
        match state.kv_storage.get_by_id(&content_key).await {
            Ok(Some(v)) => v
                .get("content")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string()),
            _ => None,
        }
    };
    match text {
        Some(text) if !text.trim().is_empty() => {
            let data = TextInsertData {
                text,
                file_source: title.clone(),
                workspace_id: workspace_id.to_string(),
                metadata: Some(serde_json::json!({
                    "document_id": document_id,
                    "title": title,
                    "source_type": if is_pdf { "pdf" } else { "markdown" },
                    "document_type": if is_pdf { "pdf" } else { "markdown" },
                    "pdf_id": pdf_id,
                    "tenant_id": tenant_id.to_string(),
                    "workspace_id": workspace_id.to_string(),
                    "is_reprocess": true,
                    // Force the full pipeline this time.
                    "skip_extraction": false,
                })),
            };
            let task = Task::new(
                tenant_id,
                workspace_id,
                TaskType::Insert,
                serde_json::to_value(data).unwrap_or_default(),
            );
            match enqueue(state, task).await {
                Ok(track_id) => queued.push(QueuedStage {
                    stage: "entities".to_string(),
                    track_id,
                }),
                Err(e) => {
                    tracing::warn!(document_id = %document_id, error = %e, "Failed to queue entity extraction")
                }
            }
        }
        _ => {
            tracing::warn!(document_id = %document_id, "No stored text for entity extraction — skipping that stage")
        }
    }

    // ── 2. Algorithm extraction ──
    let (vision_provider, vision_model) = {
        let ws = state
            .workspace_service
            .get_workspace(workspace_id)
            .await
            .ok()
            .flatten();
        (
            ws.as_ref().and_then(|w| w.vision_llm_provider.clone()),
            ws.as_ref().and_then(|w| w.vision_llm_model.clone()),
        )
    };
    let algo_data = if is_pdf {
        pdf_id.as_ref().map(|pid| AlgorithmExtractionData {
            document_id: document_id.to_string(),
            workspace_id: workspace_id.to_string(),
            chunks: vec![],
            pdf_id: Some(pid.clone()),
            vision_provider: vision_provider.clone(),
            vision_model: vision_model.clone(),
        })
    } else {
        let chunks = stitch_chunk_contents(state, document_id).await;
        if chunks.is_empty() {
            None
        } else {
            Some(AlgorithmExtractionData {
                document_id: document_id.to_string(),
                workspace_id: workspace_id.to_string(),
                chunks,
                pdf_id: None,
                vision_provider: None,
                vision_model: None,
            })
        }
    };
    if let Some(data) = algo_data {
        let task = Task::new(
            tenant_id,
            workspace_id,
            TaskType::AlgorithmExtraction,
            serde_json::to_value(data).unwrap_or_default(),
        );
        match enqueue(state, task).await {
            Ok(track_id) => queued.push(QueuedStage {
                stage: "algorithms".to_string(),
                track_id,
            }),
            Err(e) => {
                tracing::warn!(document_id = %document_id, error = %e, "Failed to queue algorithm extraction")
            }
        }
    }

    // ── 3. Reference-repo detection (PDF only) + 4. Table classification ──
    if is_pdf {
        let repo_data = RepoDetectionData {
            document_id: document_id.to_string(),
            workspace_id: workspace_id.to_string(),
            pdf_id: pdf_id.clone(),
        };
        let task = Task::new(
            tenant_id,
            workspace_id,
            TaskType::RepoDetection,
            serde_json::to_value(repo_data).unwrap_or_default(),
        );
        match enqueue(state, task).await {
            Ok(track_id) => queued.push(QueuedStage {
                stage: "reference_repos".to_string(),
                track_id,
            }),
            Err(e) => {
                tracing::warn!(document_id = %document_id, error = %e, "Failed to queue repo detection")
            }
        }

        let table_data = TableClassificationData {
            document_id: document_id.to_string(),
            workspace_id: workspace_id.to_string(),
        };
        let task = Task::new(
            tenant_id,
            workspace_id,
            TaskType::TableClassification,
            serde_json::to_value(table_data).unwrap_or_default(),
        );
        match enqueue(state, task).await {
            Ok(track_id) => queued.push(QueuedStage {
                stage: "table_classification".to_string(),
                track_id,
            }),
            Err(e) => {
                tracing::warn!(document_id = %document_id, error = %e, "Failed to queue table classification")
            }
        }
    }

    queued
}

/// Stitch a text document's stored chunk contents (`{id}-chunk-N`) into a
/// `Vec<String>` for the algorithm extractor's text path.
async fn stitch_chunk_contents(state: &AppState, document_id: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut idx = 0usize;
    loop {
        let key = format!("{}-chunk-{}", document_id, idx);
        match state.kv_storage.get_by_id(&key).await {
            Ok(Some(v)) => {
                let text = v
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_default();
                if !text.is_empty() {
                    chunks.push(text);
                }
                idx += 1;
            }
            _ => break,
        }
    }
    chunks
}

/// Load a PDF document's stored markdown (references stripped, mirroring the
/// chunking input of `process_pdf_processing`). Returns `None` when postgres
/// is unavailable or the row has no markdown.
#[cfg(feature = "postgres")]
async fn load_pdf_markdown(
    state: &AppState,
    document_id: &str,
    workspace_id: Uuid,
) -> Option<String> {
    let pool = state.pg_pool.as_ref()?;
    let doc_uuid = Uuid::parse_str(document_id).ok()?;
    let markdown: Option<String> = sqlx::query_scalar(
        r#"SELECT markdown_content FROM public.pdf_documents
           WHERE document_id = $1 AND workspace_id = $2"#,
    )
    .bind(doc_uuid)
    .bind(workspace_id)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    markdown.map(|md| edgequake_pdf::strip_references_section(&md).to_string())
}

#[cfg(not(feature = "postgres"))]
async fn load_pdf_markdown(
    _state: &AppState,
    _document_id: &str,
    _workspace_id: Uuid,
) -> Option<String> {
    None
}

/// `POST /api/v1/documents/{document_id}/extract`
///
/// Triggers the heavy LLM extraction stages for a chunks-only document.
#[utoipa::path(
    post,
    path = "/api/v1/documents/{document_id}/extract",
    params(("document_id" = String, Path, description = "Document identifier")),
    responses(
        (status = 202, description = "Extraction stages queued", body = TriggerExtractionResponse),
        (status = 400, description = "Document is not awaiting extraction"),
        (status = 404, description = "Document not found")
    ),
    tag = "Documents"
)]
pub async fn trigger_extraction(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Path(document_id): Path<String>,
) -> ApiResult<(StatusCode, Json<TriggerExtractionResponse>)> {
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

    // Gate: only documents ingested in chunks-only mode can be triggered.
    let extraction_skipped = obj
        .get("extraction_skipped")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !extraction_skipped {
        return Err(ApiError::BadRequest(
            "Document is not awaiting extraction (it was not uploaded in chunks-only mode, \
             or extraction has already been triggered)"
                .to_string(),
        ));
    }

    // Resolve tenant/workspace: prefer the document's stored ids (works for the
    // bulk path too), fall back to the caller's context.
    let workspace_id = obj
        .get("workspace_id")
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
        .or_else(|| tenant_ctx.workspace_id_uuid())
        .ok_or_else(|| ApiError::BadRequest("Could not resolve workspace_id".to_string()))?;
    let tenant_id = obj
        .get("tenant_id")
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
        .or_else(|| {
            tenant_ctx
                .tenant_id
                .as_ref()
                .and_then(|t| Uuid::parse_str(t).ok())
        })
        .ok_or_else(|| ApiError::BadRequest("Could not resolve tenant_id".to_string()))?;

    let queued =
        queue_extraction_tasks(&state, tenant_id, workspace_id, &document_id, obj).await;

    Ok((
        StatusCode::ACCEPTED,
        Json(TriggerExtractionResponse {
            document_id,
            status: "processing".to_string(),
            queued,
        }),
    ))
}
