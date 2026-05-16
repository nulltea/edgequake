//! Chunk type definition.
//!
//! A Chunk represents a segment of a document, sized appropriately for
//! LLM context windows.

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Distinguishes plain text chunks from chunks carrying inline media:
/// PDF figures (PNG image bytes) and PDF tables (rendered HTML + parsed
/// rows). The HTML / rows for tables live in dedicated columns on the
/// `chunks` table; only the kind is carried at the type level here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkKind {
    Text,
    Figure,
    Table,
}

impl Default for ChunkKind {
    fn default() -> Self {
        Self::Text
    }
}

/// Base64-encode `Option<Vec<u8>>` when writing JSON, decode on the way in.
/// Bytes-as-JSON-array is the serde default and inflates ~4×; base64 keeps the
/// KV chunk payload at the standard +33% over the raw PNG.
mod opt_bytes_b64 {
    use super::*;
    pub fn serialize<S: Serializer>(v: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(bytes) => s.serialize_str(&B64.encode(bytes)),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<u8>>, D::Error> {
        let opt: Option<String> = Option::deserialize(d)?;
        opt.map(|s| B64.decode(s.as_bytes()).map_err(serde::de::Error::custom))
            .transpose()
    }
}

/// A segment of a document.
///
/// Documents are split into chunks to fit within LLM context windows.
/// Each chunk maintains a reference back to its parent document and
/// its position within the document.
///
/// # Example
///
/// ```rust
/// use edgequake_core::types::Chunk;
///
/// let chunk = Chunk::new(
///     "This is chunk content".to_string(),
///     150,
///     0,
///     "doc-abc123".to_string(),
///     None,
/// );
/// assert!(chunk.id.starts_with("chunk-"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// MD5 hash of content - primary key
    pub id: String,
    /// Chunk text content
    pub content: String,
    /// Token count
    pub tokens: u32,
    /// Position in document (0-indexed)
    pub chunk_order_index: u32,
    /// Parent document ID
    pub full_doc_id: String,
    /// Source file path (inherited from document)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,

    // === Lineage: Position metadata ===
    // WHY: Enables tracing a chunk back to exact location in source document.
    // These fields are Optional to maintain backward compatibility with existing
    // serialized chunks that don't have position info.
    /// Start line number in source document (1-indexed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_line: Option<usize>,
    /// End line number in source document (1-indexed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    /// Start character offset in source document.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_offset: Option<usize>,
    /// End character offset in source document.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_offset: Option<usize>,

    // === Lineage: Model metadata ===
    // WHY: Enables per-chunk traceability of which LLM/embedding models were used.
    // Critical for reproducibility and quality auditing when models change over time.
    /// LLM model used for entity extraction from this chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_model: Option<String>,
    /// Embedding model used to vectorize this chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    /// Embedding vector dimension used for this chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_dimension: Option<usize>,

    // === Media payload (figure chunks) ===
    // Text chunks leave these unset; figure chunks (kind = Figure) carry the
    // PNG bytes of the cropped figure plus the stable extractor-side id that
    // links the chunk back to the markdown placeholder it replaced.
    /// What this chunk holds. Defaults to Text for backward-compatible JSON.
    #[serde(default)]
    pub kind: ChunkKind,
    /// Raw bytes of the media payload (e.g. PNG of a PDF figure crop).
    /// Encoded as base64 in JSON to keep KV chunk payloads compact.
    #[serde(default, with = "opt_bytes_b64", skip_serializing_if = "Option::is_none")]
    pub media_bytes: Option<Vec<u8>>,
    /// MIME type of `media_bytes` (e.g. `"image/png"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_mime: Option<String>,
    /// Extractor-side stable id for the figure (`fig_{page}_{order_index}`).
    /// Matches the `![figure:<id>](...)` placeholder emitted by the VLM-OCR
    /// pipeline so the chunker can pair markdown sites with figure payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub figure_id: Option<String>,
    /// Extractor-side stable id for a table chunk (`tbl_{page}_{order_index}`).
    /// Matches the `![tbl_…](edgequake-table)` placeholder. HTML and parsed
    /// rows are stored directly on the `chunks` table by the PDF processor's
    /// backfill step — not carried in this in-memory struct.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_id: Option<String>,
}

impl Chunk {
    /// Generate chunk ID from content (MD5 hash).
    ///
    /// # Example
    ///
    /// ```rust
    /// use edgequake_core::types::Chunk;
    ///
    /// let id = Chunk::generate_id("chunk content");
    /// assert!(id.starts_with("chunk-"));
    /// ```
    pub fn generate_id(content: &str) -> String {
        format!("chunk-{:x}", md5::compute(content.as_bytes()))
    }

    /// Create a new chunk.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content of the chunk
    /// * `tokens` - Number of tokens in the chunk
    /// * `chunk_order_index` - Position in the parent document (0-indexed)
    /// * `full_doc_id` - ID of the parent document
    /// * `file_path` - Optional source file path
    pub fn new(
        content: String,
        tokens: u32,
        chunk_order_index: u32,
        full_doc_id: String,
        file_path: Option<String>,
    ) -> Self {
        Self {
            id: Self::generate_id(&content),
            content,
            tokens,
            chunk_order_index,
            full_doc_id,
            file_path,
            start_line: None,
            end_line: None,
            start_offset: None,
            end_offset: None,
            llm_model: None,
            embedding_model: None,
            embedding_dimension: None,
            kind: ChunkKind::Text,
            media_bytes: None,
            media_mime: None,
            figure_id: None,
            table_id: None,
        }
    }

    /// Construct a figure chunk: caption text + image bytes + stable figure id.
    /// The id keys the chunk to the `![figure:<id>](...)` placeholder emitted
    /// by VLM-OCR, so the chunker can pair markdown sites with PNG payloads.
    pub fn new_figure(
        caption: String,
        tokens: u32,
        chunk_order_index: u32,
        full_doc_id: String,
        file_path: Option<String>,
        figure_id: String,
        media_bytes: Vec<u8>,
        media_mime: impl Into<String>,
    ) -> Self {
        // Hash the figure id into the chunk id so two figures with identical
        // captions still get distinct chunk ids.
        let id_seed = format!("figure:{figure_id}:{caption}");
        Self {
            id: Self::generate_id(&id_seed),
            content: caption,
            tokens,
            chunk_order_index,
            full_doc_id,
            file_path,
            start_line: None,
            end_line: None,
            start_offset: None,
            end_offset: None,
            llm_model: None,
            embedding_model: None,
            embedding_dimension: None,
            kind: ChunkKind::Figure,
            media_bytes: Some(media_bytes),
            media_mime: Some(media_mime.into()),
            figure_id: Some(figure_id),
            table_id: None,
        }
    }

    /// Set position metadata for lineage traceability (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `start_line` - Start line in source document (1-indexed)
    /// * `end_line` - End line in source document (1-indexed)
    /// * `start_offset` - Start character offset in source document
    /// * `end_offset` - End character offset in source document
    pub fn with_position(
        mut self,
        start_line: usize,
        end_line: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Self {
        self.start_line = Some(start_line);
        self.end_line = Some(end_line);
        self.start_offset = Some(start_offset);
        self.end_offset = Some(end_offset);
        self
    }

    /// Set model metadata for lineage traceability (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `llm_model` - LLM model used for entity extraction (e.g., "gpt-4.1-nano")
    /// * `embedding_model` - Embedding model used (e.g., "text-embedding-3-small")
    /// * `embedding_dimension` - Embedding vector dimension (e.g., 1536)
    pub fn with_models(
        mut self,
        llm_model: impl Into<String>,
        embedding_model: impl Into<String>,
        embedding_dimension: usize,
    ) -> Self {
        self.llm_model = Some(llm_model.into());
        self.embedding_model = Some(embedding_model.into());
        self.embedding_dimension = Some(embedding_dimension);
        self
    }

    /// Check if the chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty()
    }

    /// Get the content length in bytes.
    pub fn content_len(&self) -> usize {
        self.content.len()
    }

    /// Get the content length in characters.
    pub fn char_count(&self) -> usize {
        self.content.chars().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_id_generation() {
        let id1 = Chunk::generate_id("Hello chunk");
        let id2 = Chunk::generate_id("Hello chunk");
        let id3 = Chunk::generate_id("Different chunk");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert!(id1.starts_with("chunk-"));
    }

    #[test]
    fn test_chunk_creation() {
        let chunk = Chunk::new(
            "Test chunk content".to_string(),
            100,
            0,
            "doc-123".to_string(),
            Some("/test.txt".to_string()),
        );

        assert_eq!(chunk.tokens, 100);
        assert_eq!(chunk.chunk_order_index, 0);
        assert_eq!(chunk.full_doc_id, "doc-123");
        assert_eq!(chunk.file_path, Some("/test.txt".to_string()));
    }

    #[test]
    fn test_chunk_empty_check() {
        let chunk1 = Chunk::new("".to_string(), 0, 0, "doc-1".to_string(), None);
        let chunk2 = Chunk::new("   ".to_string(), 0, 0, "doc-1".to_string(), None);
        let chunk3 = Chunk::new("Content".to_string(), 10, 0, "doc-1".to_string(), None);

        assert!(chunk1.is_empty());
        assert!(chunk2.is_empty());
        assert!(!chunk3.is_empty());
    }

    #[test]
    fn test_chunk_position_default_none() {
        let chunk = Chunk::new("Content".to_string(), 10, 0, "doc-1".to_string(), None);
        assert!(chunk.start_line.is_none());
        assert!(chunk.end_line.is_none());
        assert!(chunk.start_offset.is_none());
        assert!(chunk.end_offset.is_none());
    }

    #[test]
    fn test_chunk_with_position() {
        let chunk = Chunk::new("Content".to_string(), 10, 0, "doc-1".to_string(), None)
            .with_position(1, 5, 0, 200);
        assert_eq!(chunk.start_line, Some(1));
        assert_eq!(chunk.end_line, Some(5));
        assert_eq!(chunk.start_offset, Some(0));
        assert_eq!(chunk.end_offset, Some(200));
    }

    #[test]
    fn test_chunk_position_serialization_roundtrip() {
        let chunk = Chunk::new("Content".to_string(), 10, 0, "doc-1".to_string(), None)
            .with_position(10, 20, 500, 1000);
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"start_line\":10"));
        assert!(json.contains("\"end_line\":20"));
        let deserialized: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.start_line, Some(10));
        assert_eq!(deserialized.end_offset, Some(1000));
    }

    #[test]
    fn test_chunk_backward_compat_deserialization() {
        // WHY: Existing serialized chunks without position fields must deserialize correctly.
        let old_json = r#"{"id":"chunk-abc","content":"Hello","tokens":5,"chunk_order_index":0,"full_doc_id":"doc-1"}"#;
        let chunk: Chunk = serde_json::from_str(old_json).unwrap();
        assert_eq!(chunk.content, "Hello");
        assert!(chunk.start_line.is_none());
        assert!(chunk.end_line.is_none());
        assert!(chunk.llm_model.is_none());
        assert!(chunk.embedding_model.is_none());
    }

    #[test]
    fn test_chunk_with_models() {
        let chunk = Chunk::new("Content".to_string(), 10, 0, "doc-1".to_string(), None)
            .with_models("gpt-4.1-nano", "text-embedding-3-small", 1536);
        assert_eq!(chunk.llm_model, Some("gpt-4.1-nano".to_string()));
        assert_eq!(
            chunk.embedding_model,
            Some("text-embedding-3-small".to_string())
        );
        assert_eq!(chunk.embedding_dimension, Some(1536));
    }

    #[test]
    fn test_chunk_with_full_lineage() {
        // WHY: Test that both position and model metadata can be chained.
        let chunk = Chunk::new(
            "Full lineage chunk".to_string(),
            50,
            2,
            "doc-xyz".to_string(),
            Some("/data/file.pdf".to_string()),
        )
        .with_position(10, 20, 500, 1000)
        .with_models("gpt-4.1-nano", "text-embedding-3-small", 1536);
        assert_eq!(chunk.start_line, Some(10));
        assert_eq!(chunk.llm_model, Some("gpt-4.1-nano".to_string()));
        assert_eq!(chunk.embedding_dimension, Some(1536));
        assert_eq!(chunk.full_doc_id, "doc-xyz");
        assert_eq!(chunk.file_path, Some("/data/file.pdf".to_string()));
    }

    #[test]
    fn test_chunk_model_serialization_roundtrip() {
        let chunk = Chunk::new("Content".to_string(), 10, 0, "doc-1".to_string(), None)
            .with_models("ollama/gemma3", "nomic-embed-text", 768);
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("ollama/gemma3"));
        assert!(json.contains("nomic-embed-text"));
        let deserialized: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.llm_model, Some("ollama/gemma3".to_string()));
        assert_eq!(deserialized.embedding_dimension, Some(768));
    }

    #[test]
    fn test_text_chunk_default_kind_and_no_media() {
        let chunk = Chunk::new("Content".to_string(), 10, 0, "doc-1".to_string(), None);
        assert_eq!(chunk.kind, ChunkKind::Text);
        assert!(chunk.media_bytes.is_none());
        assert!(chunk.media_mime.is_none());
        assert!(chunk.figure_id.is_none());
    }

    #[test]
    fn test_figure_chunk_constructor() {
        let png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let chunk = Chunk::new_figure(
            "Figure 2: System diagram.".to_string(),
            8,
            3,
            "doc-1".to_string(),
            None,
            "fig_4_2".to_string(),
            png.clone(),
            "image/png",
        );
        assert_eq!(chunk.kind, ChunkKind::Figure);
        assert_eq!(chunk.media_bytes.as_deref(), Some(png.as_slice()));
        assert_eq!(chunk.media_mime.as_deref(), Some("image/png"));
        assert_eq!(chunk.figure_id.as_deref(), Some("fig_4_2"));
        assert_eq!(chunk.content, "Figure 2: System diagram.");
    }

    #[test]
    fn test_figure_chunk_bytes_base64_roundtrip() {
        // Non-ASCII bytes prove we're not silently lossy-string-converting.
        let bytes: Vec<u8> = (0u8..=255).collect();
        let chunk = Chunk::new_figure(
            "caption".to_string(),
            1,
            0,
            "d".to_string(),
            None,
            "fig_0_0".to_string(),
            bytes.clone(),
            "image/png",
        );
        let json = serde_json::to_string(&chunk).unwrap();
        // Bytes must NOT appear as a JSON array of numbers.
        assert!(!json.contains("[0,1,2,3"));
        // Base64 of the all-bytes sequence starts with "AAECAw" (0,1,2,3 ...).
        assert!(json.contains("AAECAw"));
        let round: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(round.media_bytes, Some(bytes));
        assert_eq!(round.kind, ChunkKind::Figure);
    }

    #[test]
    fn test_legacy_chunk_json_deserializes_with_text_kind() {
        // WHY: existing KV blobs predate the kind/media fields. They must
        // deserialize unchanged and default to Text.
        let old = r#"{"id":"chunk-abc","content":"hi","tokens":1,"chunk_order_index":0,"full_doc_id":"d"}"#;
        let chunk: Chunk = serde_json::from_str(old).unwrap();
        assert_eq!(chunk.kind, ChunkKind::Text);
        assert!(chunk.media_bytes.is_none());
        assert!(chunk.figure_id.is_none());
    }
}
