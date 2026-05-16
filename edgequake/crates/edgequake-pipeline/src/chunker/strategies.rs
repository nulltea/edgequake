//! Chunking strategy implementations.
//!
//! Provides multiple strategies for splitting text into chunks:
//! - [`TokenBasedChunking`]: Default, token-count-based with overlap
//! - [`CharacterBasedChunking`][]: Character-count-based
//! - [`SentenceBoundaryChunking`]: Sentence-aware splitting
//! - [`ParagraphBoundaryChunking`]: Paragraph-aware splitting

use async_trait::async_trait;

use std::sync::LazyLock;

use regex::Regex;

use super::heading_path::HeadingPathIndex;
use super::text_utils::{
    estimate_tokens, split_into_sentences, split_text_internal, take_overlap_sentences,
};
use super::types::{ChunkKind, ChunkResult, ChunkerConfig, ChunkingStrategy};
use crate::error::{PipelineError, Result};

/// Matches the `![<id>](edgequake-figure)` placeholders emitted by the
/// VLM-OCR figure extractor. The id is captured in group 1.
///
/// The id pattern is the extractor's `fig_{page}_{order_index}`, conservatively
/// matched as `[A-Za-z0-9_]+` so a future id-format tweak doesn't silently
/// stop matching. The `edgequake-figure` sentinel URL is what distinguishes
/// these from regular markdown images that callers might already have written.
static FIGURE_PLACEHOLDER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!\[([A-Za-z0-9_]+)\]\(edgequake-figure\)").unwrap());

/// Matches the `![<id>](edgequake-table)` placeholders emitted by the
/// VLM-OCR table extractor. The id is captured in group 1
/// (`tbl_{page}_{order_index}`). Same conservative `[A-Za-z0-9_]+` as the
/// figure sentinel so id-format changes don't silently stop matching.
static TABLE_PLACEHOLDER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!\[([A-Za-z0-9_]+)\]\(edgequake-table\)").unwrap());

/// Default token-based chunking strategy.
///
/// This is the standard chunking strategy that splits text into chunks
/// based on token count with overlap, respecting sentence boundaries.
pub struct TokenBasedChunking;

#[async_trait]
impl ChunkingStrategy for TokenBasedChunking {
    async fn chunk(&self, content: &str, config: &ChunkerConfig) -> Result<Vec<ChunkResult>> {
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Check for split_by_character_only mode (GAP-017)
        if let Some(ref split_char) = config.split_by_character {
            if config.split_by_character_only {
                return Ok(content
                    .split(split_char.as_str())
                    .enumerate()
                    .filter(|(_, s)| !s.trim().is_empty())
                    .map(|(idx, s)| ChunkResult {
                        content: s.to_string(),
                        tokens: estimate_tokens(s),
                        chunk_order_index: idx,
                        heading_path: Vec::new(),
                        ..Default::default()
                    })
                    .collect());
            }
        }

        let target_chars = config.chunk_size * 4;
        let overlap_chars = config.chunk_overlap * 4;
        let min_chars = config.min_chunk_size * 4;

        let chunks = split_text_internal(
            content,
            target_chars,
            overlap_chars,
            min_chars,
            &config.separators,
        );

        Ok(chunks
            .into_iter()
            .enumerate()
            .map(
                |(idx, (text, _, _)): (usize, (String, usize, usize))| ChunkResult {
                    content: text.clone(),
                    tokens: estimate_tokens(&text),
                    chunk_order_index: idx,
                    heading_path: Vec::new(),
                    ..Default::default()
                },
            )
            .collect())
    }

    fn name(&self) -> &str {
        "token_based"
    }
}

/// Character-based chunking strategy (GAP-017).
///
/// Splits text on a specific character (like newline) for pre-split content.
///
/// @implements FEAT0306 (Character-Based Chunking - CharacterBasedChunking struct)
pub struct CharacterBasedChunking {
    /// Character to split on.
    pub split_character: String,
}

impl CharacterBasedChunking {
    /// Create a new character-based chunking strategy.
    pub fn new(split_character: impl Into<String>) -> Self {
        Self {
            split_character: split_character.into(),
        }
    }

    /// Create a newline-based chunker.
    pub fn by_newline() -> Self {
        Self::new("\n")
    }

    /// Create a paragraph-based chunker.
    pub fn by_paragraph() -> Self {
        Self::new("\n\n")
    }
}

#[async_trait]
impl ChunkingStrategy for CharacterBasedChunking {
    async fn chunk(&self, content: &str, _config: &ChunkerConfig) -> Result<Vec<ChunkResult>> {
        Ok(content
            .split(&self.split_character)
            .enumerate()
            .filter(|(_, s)| !s.trim().is_empty())
            .map(|(idx, s)| ChunkResult {
                content: s.to_string(),
                tokens: estimate_tokens(s),
                chunk_order_index: idx,
                heading_path: Vec::new(),
                ..Default::default()
            })
            .collect())
    }

    fn name(&self) -> &str {
        "character_based"
    }
}

/// Sentence boundary chunking strategy.
///
/// @implements SPEC-001/Issue-10: Pluggable chunk cutoff system
///
/// This strategy ensures chunks never split mid-sentence, preserving
/// complete sentences for better entity extraction context.
///
/// # Algorithm
///
/// 1. Split text into sentences using period/question/exclamation
/// 2. Accumulate sentences until target chunk size reached
/// 3. Create chunk and start new accumulation
/// 4. Overlap is handled by carrying last N sentences to next chunk
///
/// # WHY Sentence Boundaries?
///
/// Mid-sentence splits can break entity extraction context:
/// - "Dr. Smith works at Microsoft. He" → Entity "He" orphaned
/// - "Dr. Smith works at Microsoft." → Complete context preserved
pub struct SentenceBoundaryChunking;

impl SentenceBoundaryChunking {
    /// Create a new sentence boundary chunker.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SentenceBoundaryChunking {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ChunkingStrategy for SentenceBoundaryChunking {
    async fn chunk(&self, content: &str, config: &ChunkerConfig) -> Result<Vec<ChunkResult>> {
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Split into sentences (simple heuristic: period, question, exclamation)
        let sentences = split_into_sentences(content);

        if sentences.is_empty() {
            // No sentence boundaries found, fall back to token-based
            return TokenBasedChunking.chunk(content, config).await;
        }

        let target_tokens = config.chunk_size;
        let overlap_tokens = config.chunk_overlap;
        let min_tokens = config.min_chunk_size;

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_tokens = 0;
        let mut sentence_buffer: Vec<String> = Vec::new();
        let mut chunk_index = 0;

        for sentence in sentences {
            let sentence_tokens = estimate_tokens(&sentence);

            // If adding this sentence would exceed target, finalize current chunk
            if current_tokens + sentence_tokens > target_tokens && current_tokens >= min_tokens {
                chunks.push(ChunkResult {
                    content: current_chunk.trim().to_string(),
                    tokens: current_tokens,
                    chunk_order_index: chunk_index,
                    heading_path: Vec::new(),
                    ..Default::default()
                });
                chunk_index += 1;

                // Start new chunk with overlap (carry some sentences)
                let overlap_sentences = take_overlap_sentences(&sentence_buffer, overlap_tokens);
                current_chunk = overlap_sentences.join(" ");
                current_tokens = estimate_tokens(&current_chunk);
                sentence_buffer.clear();
            }

            // Add sentence to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(&sentence);
            current_tokens += sentence_tokens;
            sentence_buffer.push(sentence);
        }

        // Add final chunk if non-empty
        if current_tokens >= min_tokens {
            chunks.push(ChunkResult {
                content: current_chunk.trim().to_string(),
                tokens: current_tokens,
                chunk_order_index: chunk_index,
                heading_path: Vec::new(),
                ..Default::default()
            });
        }

        Ok(chunks)
    }

    fn name(&self) -> &str {
        "sentence_boundary"
    }
}

/// Paragraph boundary chunking strategy.
///
/// @implements SPEC-001/Issue-10: Pluggable chunk cutoff system
///
/// This strategy groups paragraphs together, never splitting within
/// a paragraph. Ideal for structured documents.
///
/// # Algorithm
///
/// 1. Split text on double newlines (paragraphs)
/// 2. Accumulate paragraphs until target chunk size reached
/// 3. Create chunk and start new accumulation
///
/// # WHY Paragraph Boundaries?
///
/// Paragraphs often contain self-contained ideas:
/// - Entity introductions usually complete within paragraph
/// - Relationships described in same paragraph as entities
/// - Splitting preserves narrative flow
pub struct ParagraphBoundaryChunking;

impl ParagraphBoundaryChunking {
    /// Create a new paragraph boundary chunker.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ParagraphBoundaryChunking {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ChunkingStrategy for ParagraphBoundaryChunking {
    async fn chunk(&self, content: &str, config: &ChunkerConfig) -> Result<Vec<ChunkResult>> {
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Split on double newlines (paragraphs)
        let paragraphs: Vec<&str> = content
            .split("\n\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if paragraphs.is_empty() {
            // No paragraphs found, try single newlines
            let single_line_paragraphs: Vec<&str> = content
                .split('\n')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            if single_line_paragraphs.is_empty() {
                return TokenBasedChunking.chunk(content, config).await;
            }
            return chunk_paragraphs(&single_line_paragraphs, config);
        }

        chunk_paragraphs(&paragraphs, config)
    }

    fn name(&self) -> &str {
        "paragraph_boundary"
    }
}

/// Helper to chunk paragraphs into size-limited chunks.
fn chunk_paragraphs(paragraphs: &[&str], config: &ChunkerConfig) -> Result<Vec<ChunkResult>> {
    let target_tokens = config.chunk_size;
    let min_tokens = config.min_chunk_size;

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_tokens = 0;
    let mut chunk_index = 0;

    for para in paragraphs {
        let para_tokens = estimate_tokens(para);

        // If this paragraph alone exceeds target, add it as its own chunk
        if para_tokens >= target_tokens {
            // First, save current accumulation if any
            if current_tokens >= min_tokens {
                chunks.push(ChunkResult {
                    content: current_chunk.trim().to_string(),
                    tokens: current_tokens,
                    chunk_order_index: chunk_index,
                    heading_path: Vec::new(),
                    ..Default::default()
                });
                chunk_index += 1;
                current_chunk = String::new();
                current_tokens = 0;
            }

            // Add large paragraph as its own chunk
            chunks.push(ChunkResult {
                content: para.to_string(),
                tokens: para_tokens,
                chunk_order_index: chunk_index,
                heading_path: Vec::new(),
                ..Default::default()
            });
            chunk_index += 1;
            continue;
        }

        // If adding would exceed target, finalize current chunk
        if current_tokens + para_tokens > target_tokens && current_tokens >= min_tokens {
            chunks.push(ChunkResult {
                content: current_chunk.trim().to_string(),
                tokens: current_tokens,
                chunk_order_index: chunk_index,
                heading_path: Vec::new(),
                ..Default::default()
            });
            chunk_index += 1;
            current_chunk = String::new();
            current_tokens = 0;
        }

        // Add paragraph to current chunk
        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(para);
        current_tokens += para_tokens;
    }

    // Add final chunk
    if current_tokens >= min_tokens {
        chunks.push(ChunkResult {
            content: current_chunk.trim().to_string(),
            tokens: current_tokens,
            chunk_order_index: chunk_index,
            heading_path: Vec::new(),
            ..Default::default()
        });
    }

    Ok(chunks)
}

// ─────────────────────────────────────────────────────────────────────────────
// Context-Aware Chunking
// ─────────────────────────────────────────────────────────────────────────────

/// Markdown-aware chunking that respects heading hierarchy and structure.
///
/// Three-stage pipeline:
///   1. Split the markdown using `text-splitter::MarkdownSplitter` sized by a
///      real BPE tokenizer (cl100k_base) — token counts within a few percent
///      of what the embedding model sees instead of the ~50% error from
///      char/4 estimation. Respects headings, lists, code fences, paragraphs.
///   2. Walk the document with [`HeadingPathIndex`] and tag each chunk with
///      its shallow-to-deep heading path.
///   3. Merge-pass: combine adjacent chunks whose combined token count is
///      still within the budget AND whose `heading_path` is identical.
///      Replaces the old `filter(|c| c.len() >= 50)` which silently dropped
///      content below the threshold.
///
/// The result is that algorithm boxes prefixed with `## Fig. N:` headings
/// stay co-located with their bodies, code fences and `$$...$$` blocks are
/// preserved atomically by the splitter, and small neighbouring sections
/// under the same heading get combined into useable chunks instead of
/// being dropped.
pub struct ContextAwareChunking;

#[async_trait]
impl ChunkingStrategy for ContextAwareChunking {
    async fn chunk(&self, content: &str, config: &ChunkerConfig) -> Result<Vec<ChunkResult>> {
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Pre-scan for figure and table placeholders. When the VLM-OCR
        // extractors captured media on this document, the markdown contains
        // one `![<id>](edgequake-figure)` or `![<id>](edgequake-table)` per
        // captured item in reading order. We split the markdown around them,
        // text-chunk each surrounding segment normally, and inject a
        // media ChunkResult at each placeholder so the persistence layer can
        // attach the image bytes / table HTML by `figure_id` / `table_id`.
        let mut placeholders: Vec<(usize, usize, String, ChunkKind)> = Vec::new();
        for cap in FIGURE_PLACEHOLDER_RE.captures_iter(content) {
            let m = cap.get(0).unwrap();
            let id = cap.get(1).unwrap().as_str().to_string();
            placeholders.push((m.start(), m.end(), id, ChunkKind::Figure));
        }
        for cap in TABLE_PLACEHOLDER_RE.captures_iter(content) {
            let m = cap.get(0).unwrap();
            let id = cap.get(1).unwrap().as_str().to_string();
            placeholders.push((m.start(), m.end(), id, ChunkKind::Table));
        }
        placeholders.sort_by_key(|(s, _, _, _)| *s);

        if placeholders.is_empty() {
            return chunk_text_only(content, config).await;
        }

        let mut out: Vec<ChunkResult> = Vec::new();
        let mut cursor = 0usize;
        let mut next_index = 0usize;
        for (start, end, id, kind) in placeholders {
            // Text segment before this placeholder.
            if start > cursor {
                let segment = &content[cursor..start];
                if !segment.trim().is_empty() {
                    let mut segs = chunk_text_only(segment, config).await?;
                    for s in segs.iter_mut() {
                        s.chunk_order_index = next_index;
                        next_index += 1;
                    }
                    out.extend(segs);
                }
            }
            // The media chunk itself. The actual payload (image bytes /
            // table HTML + parsed rows) is NOT in the markdown — it lives
            // in the PDF processor's sink and is paired by id at backfill
            // time. The placeholder text is purely a human-readable
            // anchor stored in `chunks.content`.
            let (content_placeholder, figure_id, table_id) = match kind {
                ChunkKind::Figure => (
                    format!("[figure: {id}]"),
                    Some(id.clone()),
                    None,
                ),
                ChunkKind::Table => (
                    format!("[table: {id}]"),
                    None,
                    Some(id.clone()),
                ),
                ChunkKind::Text => unreachable!("placeholder kinds are Figure or Table"),
            };
            let tokens = estimate_tokens(&content_placeholder);
            out.push(ChunkResult {
                content: content_placeholder,
                tokens,
                chunk_order_index: next_index,
                heading_path: Vec::new(),
                kind,
                figure_id,
                table_id,
                ..Default::default()
            });
            next_index += 1;
            cursor = end;
        }
        // Tail segment after the last placeholder.
        if cursor < content.len() {
            let segment = &content[cursor..];
            if !segment.trim().is_empty() {
                let mut segs = chunk_text_only(segment, config).await?;
                for s in segs.iter_mut() {
                    s.chunk_order_index = next_index;
                    next_index += 1;
                }
                out.extend(segs);
            }
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "context_aware"
    }
}

/// The original text-only ContextAwareChunking body. Kept as a free function so
/// `ContextAwareChunking::chunk` can recurse into it for the segments
/// surrounding figure placeholders. Behaviour and chunk-boundary semantics
/// are unchanged from before the placeholder split was added.
async fn chunk_text_only(content: &str, config: &ChunkerConfig) -> Result<Vec<ChunkResult>> {
    use text_splitter::{ChunkConfig, MarkdownSplitter};

    if content.trim().is_empty() {
        return Ok(Vec::new());
    }

    {
        // Real BPE tokenizer for accurate sizing. cl100k_base is reasonable
        // for any modern BPE tokenizer (Qwen, Llama, DeepSeek, …) — within
        // ~15-20% of exact; char/4 is ~50% wrong on dense technical content.
        // `CoreBPE` isn't `Clone`, so we build two — one goes into the
        // splitter as a sizer, the other is used later for token counts
        // during the merge pass.
        let sizer = tiktoken_rs::cl100k_base()
            .map_err(|e| PipelineError::ChunkingError(format!("tiktoken init: {e}")))?;
        let counter = tiktoken_rs::cl100k_base()
            .map_err(|e| PipelineError::ChunkingError(format!("tiktoken init: {e}")))?;

        // Respect the user's chunk_size verbatim (test configs use
        // intentionally tiny sizes to exercise splitting).
        let target_tokens = config.chunk_size.max(1);
        let min_tokens = config.min_chunk_size.min(target_tokens.saturating_sub(1));
        // text-splitter requires overlap < range.start (the MIN capacity),
        // not just < range.end. Cap overlap accordingly so default configs
        // like chunk_size=800/min=100/overlap=100 don't trip the check.
        let overlap_cap = if min_tokens > 0 {
            min_tokens.saturating_sub(1)
        } else {
            target_tokens.saturating_sub(1)
        };
        let overlap_tokens = config.chunk_overlap.min(overlap_cap);
        let range = if min_tokens > 0 && min_tokens < target_tokens {
            min_tokens..target_tokens
        } else {
            0..target_tokens
        };

        let splitter_config = ChunkConfig::new(range)
            .with_sizer(sizer)
            .with_overlap(overlap_tokens)
            .map_err(|e| {
                PipelineError::ChunkingError(format!(
                    "Invalid context-aware chunk config (overlap >= capacity): {e}"
                ))
            })?
            .with_trim(true);

        let splitter = MarkdownSplitter::new(splitter_config);

        // Use the indexed variant so we know each chunk's byte offset for
        // heading-path lookup.
        let indexed: Vec<(usize, &str)> = splitter.chunk_indices(content).collect();

        let heading_index = HeadingPathIndex::build(content);

        // Build pre-merge chunks with token counts and heading paths.
        #[derive(Clone)]
        struct Pending {
            content: String,
            tokens: usize,
            heading_path: Vec<String>,
        }
        let mut pending: Vec<Pending> = indexed
            .into_iter()
            .map(|(offset, slice)| {
                let tokens = counter.encode_ordinary(slice).len();
                let heading_path = heading_index.path_at(offset);
                Pending {
                    content: slice.to_string(),
                    tokens,
                    heading_path,
                }
            })
            .collect();

        // Merge-pass. Walk the vector and greedily combine chunk[i] with
        // chunk[i+1] when:
        //   - current chunk is under `min_chunk_size` OR under the target,
        //   - combined size would still fit in the target,
        //   - heading paths match exactly.
        // This fixes the old "drop chunks < 50 chars" bug (which lost
        // content) and fills the gap between MarkdownSplitter boundary
        // preservation (which can emit tiny heading-only chunks) and
        // useful-for-embedding sizing.
        let mut merged: Vec<Pending> = Vec::with_capacity(pending.len());
        let min = config.min_chunk_size;
        let max = target_tokens;
        while !pending.is_empty() {
            let mut cur = pending.remove(0);
            while let Some(next) = pending.first() {
                let combined = cur.tokens + next.tokens;
                // Only merge if (a) current is too small or already under
                // target, (b) combined stays within target, (c) both are
                // in the same section.
                let should_merge = (cur.tokens < min || combined <= max)
                    && combined <= max
                    && cur.heading_path == next.heading_path;
                if !should_merge {
                    break;
                }
                // Pop the peeked element and glue with a blank line so the
                // markdown structure is preserved.
                let n = pending.remove(0);
                cur.content.push_str("\n\n");
                cur.content.push_str(&n.content);
                cur.tokens = counter.encode_ordinary(&cur.content).len();
            }
            merged.push(cur);
        }

        Ok(merged
            .into_iter()
            .enumerate()
            .map(|(idx, p)| ChunkResult {
                content: p.content,
                tokens: p.tokens,
                chunk_order_index: idx,
                heading_path: p.heading_path,
                ..Default::default()
            })
            .collect())
    }
}

#[cfg(test)]
mod figure_placeholder_tests {
    use super::*;

    #[tokio::test]
    async fn no_placeholders_means_no_figure_chunks() {
        let md = "# Heading\n\nParagraph one. Paragraph two has more content.";
        let chunks = ContextAwareChunking
            .chunk(md, &ChunkerConfig::default())
            .await
            .unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.kind == ChunkKind::Text));
        assert!(chunks.iter().all(|c| c.figure_id.is_none()));
    }

    #[tokio::test]
    async fn single_placeholder_emits_one_figure_chunk() {
        let md = "Intro paragraph.\n\n![fig_2_5](edgequake-figure)\n\nFollow-up paragraph.";
        let chunks = ContextAwareChunking
            .chunk(md, &ChunkerConfig::default())
            .await
            .unwrap();
        let figs: Vec<&ChunkResult> = chunks.iter().filter(|c| c.kind == ChunkKind::Figure).collect();
        assert_eq!(figs.len(), 1);
        assert_eq!(figs[0].figure_id.as_deref(), Some("fig_2_5"));
        // The figure chunk must sit between the surrounding text chunks in order.
        let positions: Vec<usize> = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.kind == ChunkKind::Figure)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(positions.len(), 1);
        // Should not be first or last in a typical multi-segment doc.
        assert!(positions[0] > 0);
        assert!(positions[0] < chunks.len() - 1);
        // Continuous order indices.
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_order_index, i);
        }
    }

    #[tokio::test]
    async fn multiple_placeholders_keep_reading_order() {
        let md = "A\n\n![fig_1_3](edgequake-figure)\n\nB\n\n![fig_2_7](edgequake-figure)\n\nC";
        let chunks = ContextAwareChunking
            .chunk(md, &ChunkerConfig::default())
            .await
            .unwrap();
        let figs: Vec<&ChunkResult> = chunks.iter().filter(|c| c.kind == ChunkKind::Figure).collect();
        assert_eq!(figs.len(), 2);
        assert_eq!(figs[0].figure_id.as_deref(), Some("fig_1_3"));
        assert_eq!(figs[1].figure_id.as_deref(), Some("fig_2_7"));
        // Continuous order indices across the whole sequence.
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_order_index, i);
        }
    }

    #[tokio::test]
    async fn placeholder_at_end_with_no_trailing_text() {
        let md = "Some intro text here.\n\n![fig_0_0](edgequake-figure)";
        let chunks = ContextAwareChunking
            .chunk(md, &ChunkerConfig::default())
            .await
            .unwrap();
        assert!(chunks.last().unwrap().kind == ChunkKind::Figure);
        assert_eq!(
            chunks.last().unwrap().figure_id.as_deref(),
            Some("fig_0_0")
        );
    }

    #[tokio::test]
    async fn regular_markdown_images_are_not_misclassified() {
        // Non-sentinel URL: this is a real image link, not a figure placeholder.
        let md = "Intro.\n\n![alt text](https://example.com/img.png)\n\nMore.";
        let chunks = ContextAwareChunking
            .chunk(md, &ChunkerConfig::default())
            .await
            .unwrap();
        assert!(chunks.iter().all(|c| c.kind == ChunkKind::Text));
    }
}
