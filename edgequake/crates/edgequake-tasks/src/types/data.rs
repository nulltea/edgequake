//! Task-specific data payloads.
//!
//! Typed payloads for each task type, serialized into the
//! `task_data` JSON field of a Task.

use edgequake_pdf::PdfParserBackend;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Document upload task payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentUploadData {
    pub file_path: String,
    pub content_type: String,
    pub workspace_id: String,
    pub metadata: Option<serde_json::Value>,
}

/// PDF processing task payload
///
/// @implements SPEC-007: PDF Upload Support
///
/// This structure contains all information needed to process a PDF document:
/// - Extract content (text or vision)
/// - Convert to markdown
/// - Ingest into knowledge graph
///
/// @implements SPEC-002: Unified Ingestion Pipeline
/// OODA-05: Added tenant_id for multi-tenant context propagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfProcessingData {
    /// PDF document ID
    pub pdf_id: Uuid,

    /// Tenant ID for multi-tenant isolation
    /// OODA-05: Required for document metadata to be visible in workspace queries
    pub tenant_id: Uuid,

    /// Workspace ID for isolation
    pub workspace_id: Uuid,

    /// Enable vision LLM processing
    pub enable_vision: bool,

    /// Vision provider to use (openai, ollama)
    pub vision_provider: String,

    /// Optional vision model override
    pub vision_model: Option<String>,

    /// Existing document ID to reuse during rebuild/reprocessing.
    /// WHY: When rebuilding knowledge graph or reprocessing PDF documents,
    /// we must reuse the existing document ID so the old document is updated
    /// in-place rather than creating an orphaned duplicate. Without this,
    /// the old document still references the same pdf_id whose markdown_content
    /// was overwritten, causing it to display wrong/hallucinated content.
    #[serde(default)]
    pub existing_document_id: Option<String>,

    /// PDF parser backend to use for this task.
    /// Old queued tasks omit this field and therefore default to Vision.
    #[serde(default)]
    pub pdf_parser_backend: PdfParserBackend,

    /// When `true`, the post-OCR stage will attempt to rename the PDF
    /// to a citation-style filename (`"Author et al. - Year - Title.pdf"`)
    /// using front-matter parsed from the extracted markdown. Silently
    /// skipped when front-matter isn't extractable. Set by the
    /// `/documents/pdf/from-url` endpoint for URL-initiated uploads;
    /// file uploads default to `false` so user-chosen filenames survive.
    #[serde(default)]
    pub rename_after_parse: bool,

    /// Source URL the PDF was fetched from (only set for URL uploads).
    /// The rename step parses the arxiv-id regex off this to derive a
    /// publication year when the URL points to arxiv.
    #[serde(default)]
    pub source_url: Option<String>,
}

/// Text insert task payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextInsertData {
    pub text: String,
    pub file_source: String,
    pub workspace_id: String,
    pub metadata: Option<serde_json::Value>,
}

/// Directory scan task payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryScanData {
    pub directory_path: String,
    pub recursive: bool,
    pub file_pattern: Option<String>,
    pub workspace_id: String,
}

/// Algorithm extraction task payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmExtractionData {
    pub document_id: String,
    pub workspace_id: String,
    /// Document chunks (text documents only, fallback path).
    /// For PDFs, `pdf_id` is set and layout detection is used instead.
    pub chunks: Vec<String>,
    /// If set, the processor uses layout detection + VLM to find algorithm blocks
    /// directly from the PDF pages instead of scanning text chunks.
    #[serde(default)]
    pub pdf_id: Option<String>,
    /// Vision provider name for VLM recognition (e.g. "openai-compatible").
    #[serde(default)]
    pub vision_provider: Option<String>,
    /// Vision model name for VLM recognition (e.g. "GLM-OCR").
    #[serde(default)]
    pub vision_model: Option<String>,
}

/// Algorithm embedding task payload — generates vector embeddings for approved algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmEmbeddingData {
    pub document_id: String,
    pub workspace_id: String,
    /// Algorithm IDs to embed (all approved algorithms for this document).
    pub algorithm_ids: Vec<String>,
}

/// Reference-repository detection task payload (Phase 0 of the Reference
/// Code GraphRAG extension). Runs Layer A (PDF hyperlinks) and, if nothing
/// is found, Layer B (SearXNG + Crawl4AI web-search fallback).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoDetectionData {
    pub document_id: String,
    pub workspace_id: String,
    /// PDF id. Set when the document was ingested as a PDF — enables Layer A.
    /// If None, only Layer B runs (assumes the markdown is stored and loadable).
    #[serde(default)]
    pub pdf_id: Option<String>,
}

/// Reference-code analysis task payload (Phase 1 of the Reference Code
/// GraphRAG extension). Kicked off when a user approves a `document_repos`
/// row — edgequake will call the code-analyzer sidecar, locate approved
/// algorithms inside the repo, and persist `code_artifacts` for review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeReferenceAnalysisData {
    pub document_id: String,
    pub workspace_id: String,
    /// The approved document_repos row we're about to analyze.
    pub document_repo_id: uuid::Uuid,
}

/// Full reference-codebase indexing task payload (Phase 2 of the Reference
/// Code GraphRAG extension). Uses the persisted clone created by the
/// `code-analyzer` sidecar and builds a separate codebase RAG index for
/// coding-agent retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceCodebaseIndexData {
    pub document_id: String,
    pub workspace_id: String,
    pub document_repo_id: uuid::Uuid,
    /// Indexing mode: "algorithm_focused" (default) or "full".
    #[serde(default = "default_reference_codebase_mode")]
    pub mode: String,
    /// Rebuild even when an index already exists for the same repo commit.
    #[serde(default)]
    pub force_reindex: bool,
}

fn default_reference_codebase_mode() -> String {
    "algorithm_focused".to_string()
}

/// Reindex task payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReindexData {
    pub document_ids: Vec<String>,
    pub workspace_id: String,
    pub reason: String,
}
