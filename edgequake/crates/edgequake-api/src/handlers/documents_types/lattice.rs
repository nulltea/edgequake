//! Plan-6 lattice-facing DTOs.
//!
//! `POST /api/v1/documents/raw`           — raw kv_storage write, no extraction
//! `POST /api/v1/documents/{id}/chunks`   — pre-chunked array, server embeds
//! `PATCH /api/v1/documents/{id}`         — status update
//!
//! These exist because lattice owns its own AI extraction + chunking and only
//! needs edgequake to be storage + retrieval. The regular upload_document
//! path runs the full ingestion pipeline (chunking + LLM extraction +
//! embedding); these endpoints skip everything except the persistence step
//! so lattice's writes don't trigger duplicate-extraction.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `POST /documents/raw` request.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct RawDocumentRequest {
    /// Document body (markdown). Stored in `kv_storage` so it shows up on the
    /// /documents page, but never chunked or extracted by edgequake.
    pub content: String,
    /// Optional title for the documents-page display.
    #[serde(default)]
    pub title: Option<String>,
    /// Source identifier (typically a relative path) lattice uses to dedupe.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Initial status to record. Lattice usually starts at "processing" and
    /// PATCHes to "completed" once its own extraction finishes.
    #[serde(default)]
    pub status: Option<String>,
}

/// `POST /documents/raw` response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct RawDocumentResponse {
    pub document_id: String,
    pub status: String,
}

/// One pre-chunked unit lattice POSTs.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct ChunkUpload {
    /// Stable chunk identifier (e.g. `research/foo.md@chunk-3`). Used as the
    /// vector-storage key.
    pub id: String,
    /// Chunk body. Stored on the vector row as `metadata.content`.
    pub content: String,
    /// Whether the server should embed this chunk via the workspace's
    /// configured embedding provider and store the vector. Default false.
    #[serde(default)]
    pub embed: Option<bool>,
    /// Optional override for the embedding input. Falls back to `content`.
    #[serde(default)]
    pub embedding_text: Option<String>,
    /// Free-form per-chunk metadata. Merged with the server-set
    /// `{type:"chunk", document_id, workspace_id, tenant_id}`.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// `POST /documents/{id}/chunks` request.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct CreateChunksRequest {
    pub chunks: Vec<ChunkUpload>,
}

/// `POST /documents/{id}/chunks` response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct CreateChunksResponse {
    pub document_id: String,
    pub chunks_stored: usize,
    pub embeddings_generated: usize,
}

/// `PATCH /documents/{id}` request.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct UpdateDocumentRequest {
    /// New processing status (`pending` / `processing` / `completed` /
    /// `failed`).
    #[serde(default)]
    pub status: Option<String>,
    /// Optional stage label for surfacing more granular progress.
    #[serde(default)]
    pub current_stage: Option<String>,
    /// Optional human-readable progress message.
    #[serde(default)]
    pub stage_message: Option<String>,
    /// Number of entities lattice extracted for this doc — surfaced in
    /// the /documents page.
    #[serde(default)]
    pub entity_count: Option<usize>,
    /// Number of relationships lattice wrote for this doc.
    #[serde(default)]
    pub relationship_count: Option<usize>,
    /// LLM token usage for the extraction pass (sum across turns).
    #[serde(default)]
    pub input_tokens: Option<usize>,
    #[serde(default)]
    pub output_tokens: Option<usize>,
    #[serde(default)]
    pub total_tokens: Option<usize>,
    /// USD cost of the extraction pass (SDK-reported).
    #[serde(default)]
    pub cost_usd: Option<f64>,
    /// Model name used for extraction (e.g. `claude-haiku-4-5-20251001`).
    #[serde(default)]
    pub llm_model: Option<String>,
    /// Embedding model used for the workspace at sync time.
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// Wall-clock processing duration in milliseconds (extraction + writes).
    #[serde(default)]
    pub processing_duration_ms: Option<u64>,
}

/// `PATCH /documents/{id}` response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct UpdateDocumentResponse {
    pub document_id: String,
    pub status: String,
}
