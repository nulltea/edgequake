//! Core types and traits for text chunking.
//!
//! Defines the data structures and strategy trait used by the chunker module.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Distinguishes plain text chunks from figure chunks (PNG image payload) and
/// table chunks (HTML + parsed-rows payload). Mirrors
/// `edgequake_core::types::ChunkKind` so the chunker can surface the same
/// distinction at the strategy boundary.
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

/// Result of a custom chunking operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkResult {
    /// The chunk text content.
    pub content: String,
    /// Approximate token count.
    pub tokens: usize,
    /// Zero-based index indicating the chunk's order in the document.
    pub chunk_order_index: usize,
    /// Heading hierarchy active at the chunk's location, shallow-to-deep
    /// (e.g. `["II. Preliminaries", "B. B2A Protocol"]`). Empty when the
    /// chunk sits before any heading or when the strategy doesn't track
    /// paths. Used by merge logic and by retrieval to prefer in-section
    /// matches.
    #[serde(default)]
    pub heading_path: Vec<String>,
    /// Whether this is a plain text chunk or a figure chunk carrying media.
    #[serde(default)]
    pub kind: ChunkKind,
    /// PNG (or other media) bytes for figure chunks. None for text chunks.
    /// Stored uncompressed in memory; the persistence layer base64-encodes
    /// when writing to the KV JSON path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_bytes: Option<Vec<u8>>,
    /// MIME type of `media_bytes` (e.g. `"image/png"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_mime: Option<String>,
    /// Extractor-side stable id (`fig_{page}_{order_index}`) linking this
    /// chunk to the `![figure:<id>](...)` markdown placeholder it replaced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub figure_id: Option<String>,
    /// Extractor-side stable id (`tbl_{page}_{order_index}`) linking this
    /// chunk to the `![tbl_…](edgequake-table)` placeholder it replaced.
    /// The HTML and parsed rows live in `chunks.table_html` / `table_rows`
    /// and get written by the PDF processor's backfill step, not here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_id: Option<String>,
}

/// Trait for custom chunking strategies.
///
/// Implement this trait to provide your own chunking logic for document processing.
/// This allows for flexible chunking strategies such as:
/// - Semantic chunking (based on meaning/topics)
/// - Fixed-size chunking with custom separators
/// - Language-specific chunking (code, markdown, etc.)
#[async_trait]
pub trait ChunkingStrategy: Send + Sync {
    /// Chunk the given text content into smaller pieces.
    ///
    /// # Arguments
    /// * `content` - The full text content to chunk
    /// * `config` - The chunking configuration
    ///
    /// # Returns
    /// A vector of chunk results with content, token count, and order index
    async fn chunk(&self, content: &str, config: &ChunkerConfig) -> Result<Vec<ChunkResult>>;

    /// Get the name of this chunking strategy.
    fn name(&self) -> &str;
}

/// Configuration for the chunker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkerConfig {
    /// Target chunk size in tokens.
    pub chunk_size: usize,

    /// Overlap between chunks in tokens.
    pub chunk_overlap: usize,

    /// Minimum chunk size (won't create chunks smaller than this).
    pub min_chunk_size: usize,

    /// Separator characters for splitting.
    pub separators: Vec<String>,

    /// Whether to preserve sentence boundaries.
    pub preserve_sentences: bool,

    /// Optional character to split on first (e.g., "\n" for line-by-line).
    pub split_by_character: Option<String>,

    /// If true, split only on the specified character, don't apply token limits.
    pub split_by_character_only: bool,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            // Tier B: ContextAwareChunking counts real cl100k BPE tokens, so these
            // values are in real tokens (not the old char/4 estimate). qwen3-embedding
            // handles 32K, so the ceiling isn't the constraint — retrieval quality is.
            // 1600 tokens ≈ 3200–4800 chars depending on text density, roughly matching
            // the old TokenBasedChunking effective size. min_chunk_size=600 forces the
            // merge pass to aggressively collapse leaf-subsection stubs; without it a
            // paper with ~20 heading anchors fragments into ~45 chunks.
            chunk_size: 1600,
            chunk_overlap: 100,
            min_chunk_size: 600,
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                "! ".to_string(),
                "? ".to_string(),
                "; ".to_string(),
                ", ".to_string(),
                " ".to_string(),
            ],
            preserve_sentences: true,
            split_by_character: None,
            split_by_character_only: false,
        }
    }
}

/// A chunk of text with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    /// Unique identifier for the chunk.
    pub id: String,

    /// The chunk text content.
    pub content: String,

    /// Index of this chunk in the document.
    pub index: usize,

    /// Character offset from the start of the document.
    pub start_offset: usize,

    /// Character offset to the end of the chunk.
    pub end_offset: usize,

    /// Starting line number (1-based) in the original document.
    pub start_line: usize,

    /// Ending line number (1-based, inclusive) in the original document.
    pub end_line: usize,

    /// Approximate token count.
    pub token_count: usize,

    /// Heading hierarchy active at this chunk's location, shallow-to-deep.
    /// Empty when the chunk sits before any heading. Populated by
    /// `ContextAwareChunking`; other strategies currently leave this empty.
    #[serde(default)]
    pub heading_path: Vec<String>,

    /// Chunk embedding.
    pub embedding: Option<Vec<f32>>,

    /// Whether this is a plain text chunk or a figure chunk carrying media.
    /// Figure chunks are emitted by `ContextAwareChunking` when it encounters
    /// `![<id>](edgequake-figure)` placeholders left by the VLM-OCR figure
    /// extractor in the markdown.
    #[serde(default)]
    pub kind: ChunkKind,
    /// PNG (or other media) bytes for figure chunks. None at chunker output —
    /// the bytes live in the PDF processor's sink and are attached to the
    /// chunk record at persistence time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_bytes: Option<Vec<u8>>,
    /// MIME type of `media_bytes` (e.g. `"image/png"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_mime: Option<String>,
    /// Stable extractor-side figure id (`fig_{page}_{order_index}`). The PDF
    /// processor uses this to pair the chunk row with the PNG payload sitting
    /// in `extracted_figures`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub figure_id: Option<String>,
    /// Stable extractor-side table id (`tbl_{page}_{order_index}`) for table
    /// chunks emitted at `![tbl_…](edgequake-table)` placeholder sites.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_id: Option<String>,
}

impl TextChunk {
    /// Create a new text chunk.
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        index: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Self {
        let content = content.into();
        let token_count = super::text_utils::estimate_tokens(&content);
        Self {
            id: id.into(),
            content,
            index,
            start_offset,
            end_offset,
            start_line: 1,
            end_line: 1,
            token_count,
            heading_path: Vec::new(),
            embedding: None,
            kind: ChunkKind::Text,
            media_bytes: None,
            media_mime: None,
            figure_id: None,
            table_id: None,
        }
    }

    /// Create a new text chunk with line numbers.
    pub fn with_line_numbers(
        id: impl Into<String>,
        content: impl Into<String>,
        index: usize,
        start_offset: usize,
        end_offset: usize,
        start_line: usize,
        end_line: usize,
    ) -> Self {
        let content = content.into();
        let token_count = super::text_utils::estimate_tokens(&content);
        Self {
            id: id.into(),
            content,
            index,
            start_offset,
            end_offset,
            start_line,
            end_line,
            token_count,
            heading_path: Vec::new(),
            embedding: None,
            kind: ChunkKind::Text,
            media_bytes: None,
            media_mime: None,
            figure_id: None,
            table_id: None,
        }
    }

    /// Set the heading path after creation.
    pub fn set_heading_path(&mut self, heading_path: Vec<String>) {
        self.heading_path = heading_path;
    }

    /// Set line numbers after creation.
    pub fn set_line_numbers(&mut self, start_line: usize, end_line: usize) {
        self.start_line = start_line;
        self.end_line = end_line;
    }
}
