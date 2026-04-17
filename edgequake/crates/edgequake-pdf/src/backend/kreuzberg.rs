//! High-fidelity PDF → Markdown backend powered by Kreuzberg.
//!
//! Uses the [`kreuzberg`](https://crates.io/crates/kreuzberg) document extraction
//! library with layout detection (RT-DETR v2) and TATR table reconstruction for
//! best quality on academic papers.
//!
//! Key quality levers configured here (per Kreuzberg docs tuning for scientific PDFs):
//! - **Layout detection** — classifies regions as SectionHeader / Formula / Table / Text / PageHeader.
//!   Fixes author-affiliation being promoted to H2, section titles being inlined as paragraphs,
//!   and equation fragments becoming headings.
//! - **TATR table model** — structural table reconstruction (not heuristic), so cost-comparison
//!   tables come out as proper markdown tables instead of list items.
//! - **Hierarchy with 5 font-size clusters** — fits IEEE/ACM templates (Title + Section +
//!   Subsection + Subsubsection + Body).
//! - **Content filter** — strips running headers, page numbers, arXiv watermarks.
//! - **Explicit margin fractions** — clips headers/footers before classification.
//!
//! OCR is disabled here — scanned/image-only PDFs should use the `Vision` backend.

use async_trait::async_trait;
use kreuzberg::core::config::{
    ContentFilterConfig, ExtractionConfig, HierarchyConfig, LayoutDetectionConfig, OutputFormat,
    PdfBackend, PdfConfig, TableModel,
};
use kreuzberg::extract_bytes;
use tracing::{info, warn};

use super::{PdfConversionConfig, PdfConverter};
use crate::error::PdfConversionError;

/// Kreuzberg-powered PDF → markdown converter (CPU-only, no LLM, no OCR).
#[derive(Debug, Default)]
pub struct KreuzbergConverter;

#[async_trait]
impl PdfConverter for KreuzbergConverter {
    async fn convert(
        &self,
        pdf_bytes: &[u8],
        _config: &PdfConversionConfig,
    ) -> Result<String, PdfConversionError> {
        let mut cfg = ExtractionConfig::default();

        // Output markdown (default is Plain; we want structured markdown).
        cfg.output_format = OutputFormat::Markdown;
        cfg.include_document_structure = true;
        cfg.enable_quality_processing = true; // NFC + mojibake fixup
        cfg.use_cache = false;

        // Layout detection — RT-DETR v2 region classification. The single biggest quality lever.
        // Fixes mis-promoted headings, inlined section titles, and enables proper formula/table tagging.
        // TATR (Table Transformer) reconstructs table structure; swap to SlanetWired for bordered tables.
        cfg.layout = Some(LayoutDetectionConfig {
            confidence_threshold: Some(0.5),
            apply_heuristics: true,
            table_model: TableModel::Tatr,
        });

        // PDF-specific tuning for academic papers.
        cfg.pdf_options = Some(PdfConfig {
            backend: PdfBackend::Pdfium,
            extract_metadata: true,
            extract_images: false,
            extract_annotations: false,
            hierarchy: Some(HierarchyConfig {
                enabled: true,
                // 5 clusters fits IEEE/ACM templates (Title + Section + Subsection + Subsubsection + Body).
                k_clusters: 5,
                include_bbox: true,
                ocr_coverage_threshold: None,
            }),
            // Clip running headers/footers before classification (explicit defaults).
            top_margin_fraction: Some(0.06),
            bottom_margin_fraction: Some(0.05),
            allow_single_column_tables: false,
            passwords: None,
        });

        // Strip conference running heads, page numbers, arXiv watermark.
        cfg.content_filter = Some(ContentFilterConfig {
            include_headers: false,
            include_footers: false,
            strip_repeating_text: true,
            include_watermarks: false,
        });

        // Explicitly disable OCR — use the Vision backend for scanned PDFs.
        cfg.disable_ocr = true;

        let result = extract_bytes(pdf_bytes, "application/pdf", &cfg)
            .await
            .map_err(|e| {
                PdfConversionError::Backend(format!("Kreuzberg extraction failed: {e}"))
            })?;

        let markdown = result.content;

        if markdown.trim().is_empty() {
            return Err(PdfConversionError::EmptyOutput(
                "Kreuzberg returned no text — PDF may be scanned or encrypted",
            ));
        }

        let trimmed_len = markdown.trim().len();
        if trimmed_len < 200 {
            warn!(
                markdown_len = trimmed_len,
                "Low text content from Kreuzberg backend — PDF may be scanned/image-only"
            );
        }

        // Post-process: rescue boxed algorithm/protocol/functionality blocks that the
        // layout model misclassifies as flat list items. See module-level NOTE below.
        let reflowed = postprocess_algorithm_blocks(&markdown);

        info!(
            markdown_len = reflowed.len(),
            original_len = markdown.len(),
            table_count = result.tables.len(),
            "Kreuzberg conversion completed"
        );

        Ok(reflowed)
    }

    fn backend_name(&self) -> &'static str {
        "kreuzberg"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-processing — rescue boxed algorithm blocks
// ─────────────────────────────────────────────────────────────────────────────
//
// NOTE: Kreuzberg's RT-DETR layout model (trained on DocLayNet) has no class for
// visually-boxed content like IEEE "Protocol 1 Π1" or "Functionality FCR" frames.
// Such blocks come out as flat `- ` bullets with all steps concatenated on one line:
//
//   - Functionality: JbK A ← Bit2A(JbKB, λ) Input: A Boolean secret sharing JbK B
//     where b ∈ {0, 1}. Output: An arithmetical secret sharing JbK A ... Setup phase:
//     1: For each i ∈ {0, 1, 2}, Pi samples random seeds Ri ∈ {0, 1} λ and share it
//     with Pi−1. 2: P0 and P1: α1 ← Prg(R1, ℓ) 3: P0: γ ...
//
// This post-processor reshapes such blocks:
//   - Promotes "Protocol N ..." / "Functionality F..." / "Algorithm N..." to `####` heading
//   - Bolds structural labels (Input: Output: Setup phase: Online phase: Functionality:)
//   - Splits inline-concatenated numbered steps into a proper numbered list
//
// This is a deterministic, regex-free, no-new-deps fix. Upstream (kreuzberg-dev/kreuzberg)
// has no pending work on this, so we solve it locally. See research notes in
// /home/timo/.claude/plans/mossy-whistling-horizon.md.

/// Headings that introduce a boxed algorithm/protocol/functionality block.
const ALGO_BLOCK_HEADINGS: &[&str] = &[
    "Protocol",
    "Functionality",
    "Algorithm",
    "Definition",
    "Theorem",
    "Lemma",
    "Proposition",
    "Corollary",
];

/// Structural labels inside a boxed block that should be bolded.
const ALGO_BLOCK_LABELS: &[&str] = &[
    "Input:",
    "Output:",
    "Setup phase:",
    "Online phase:",
    "Setup:",
    "Online:",
    "Functionality:",
    "Requires:",
    "Ensures:",
    "Precondition:",
    "Postcondition:",
];

fn postprocess_algorithm_blocks(markdown: &str) -> String {
    let mut out = String::with_capacity(markdown.len() + 512);

    for line in markdown.lines() {
        let trimmed = line.trim_start();
        // Match bullet-list items: `- X` or `* X` where X starts with an algo-heading keyword
        // followed by a digit (e.g., "Protocol 1", "Functionality 3"), OR a label keyword
        // like "Functionality: Bit2A(...)".
        let bullet_content = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "));

        let is_algo_block = bullet_content
            .map(|rest| starts_with_algo_heading(rest))
            .unwrap_or(false);

        if is_algo_block {
            let content = bullet_content.unwrap();
            let reflowed = reflow_algo_block(content);
            out.push_str(&reflowed);
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    out
}

/// Check if a line starts with an algorithm-block heading keyword followed by a
/// digit, colon, or paren — signalling a boxed block we want to rescue.
fn starts_with_algo_heading(text: &str) -> bool {
    for kw in ALGO_BLOCK_HEADINGS {
        if let Some(rest) = text.strip_prefix(kw) {
            // "Protocol 1 ..." — digit follows; "Functionality:" — colon follows
            let next = rest.chars().next();
            match next {
                Some(c) if c.is_whitespace() => {
                    // "Protocol " — check what follows the space
                    let after = rest.trim_start();
                    if after.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        return true;
                    }
                }
                Some(':') => return true,
                _ => {}
            }
        }
    }
    false
}

/// Reflow a single bulleted algorithm block: promote heading, bold labels, split steps.
fn reflow_algo_block(content: &str) -> String {
    let mut out = String::with_capacity(content.len() + 256);

    // Split off the heading line: everything up to the first label or " 1: " step.
    let (heading, body) = split_heading_from_body(content);

    // Emit heading as H4 (#### fits well under existing H2/H3 section structure)
    out.push_str("\n#### ");
    out.push_str(heading.trim());
    out.push_str("\n\n");

    if body.is_empty() {
        return out;
    }

    // Split the body on structural labels AND numbered steps.
    let segments = split_into_segments(body);

    let mut in_list = false;
    let mut step_num: usize = 1;

    for seg in segments {
        let seg = seg.trim();
        if seg.is_empty() {
            continue;
        }

        // Is this a step (starts with a number followed by `:` or `.`)?
        if let Some((_n, rest)) = extract_leading_step_number(seg) {
            if !in_list {
                in_list = true;
                out.push('\n');
            }
            out.push_str(&format!("{}. {}\n", step_num, rest.trim()));
            step_num += 1;
            continue;
        }

        // Is this a label (Input: / Output: / Setup phase: etc)?
        if let Some((label, rest)) = extract_leading_label(seg) {
            if in_list {
                out.push('\n');
                in_list = false;
            }
            let rest = rest.trim();
            if rest.is_empty() {
                out.push_str(&format!("**{}**\n\n", label));
            } else {
                out.push_str(&format!("**{}** {}\n\n", label, rest));
            }
            continue;
        }

        // Plain paragraph
        if in_list {
            out.push('\n');
            in_list = false;
        }
        out.push_str(seg);
        out.push_str("\n\n");
    }

    out
}

/// Split the content into (heading, body). Heading is everything up to the first
/// label (`Input:`, `Output:`, etc.) or the first step (` 1: ` / ` 1. `).
fn split_heading_from_body(content: &str) -> (&str, &str) {
    // Find the earliest match of any label or a step marker " N:" / " N."
    let mut earliest: Option<usize> = None;

    for label in ALGO_BLOCK_LABELS {
        if let Some(pos) = find_word_boundary(content, label) {
            earliest = Some(earliest.map_or(pos, |e| e.min(pos)));
        }
    }

    // Also find first occurrence of " N:" or " N." where N is a digit (step marker).
    let bytes = content.as_bytes();
    let mut i = 0;
    while i + 2 < bytes.len() {
        if bytes[i] == b' '
            && bytes[i + 1].is_ascii_digit()
            && (bytes[i + 2] == b':' || bytes[i + 2] == b'.')
        {
            // Ensure the digit is surrounded by plausible step-like context
            let pos = i + 1; // skip the leading space — step starts at the digit
            earliest = Some(earliest.map_or(pos, |e| e.min(pos)));
            break;
        }
        i += 1;
    }

    match earliest {
        Some(pos) => (&content[..pos], &content[pos..]),
        None => (content, ""),
    }
}

/// Find the byte position of `needle` in `haystack` only at a word boundary (preceded by whitespace or start).
fn find_word_boundary(haystack: &str, needle: &str) -> Option<usize> {
    let bytes = haystack.as_bytes();
    let nb = needle.as_bytes();
    if nb.is_empty() || bytes.len() < nb.len() {
        return None;
    }
    let mut i = 0;
    while i + nb.len() <= bytes.len() {
        if &bytes[i..i + nb.len()] == nb {
            let prev_is_boundary = i == 0 || bytes[i - 1].is_ascii_whitespace() || bytes[i - 1] == b'.';
            if prev_is_boundary {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Split the body by structural markers: step numbers and labels.
fn split_into_segments(body: &str) -> Vec<String> {
    let mut segments: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut words_iter = body.split_ascii_whitespace().peekable();

    while let Some(word) = words_iter.next() {
        // Is this the start of a new labelled segment?
        let is_label = ALGO_BLOCK_LABELS.iter().any(|l| word == *l);
        let is_step = looks_like_step_marker(word);

        if (is_label || is_step) && !current.trim().is_empty() {
            segments.push(std::mem::take(&mut current));
        }

        current.push_str(word);
        current.push(' ');
    }

    if !current.trim().is_empty() {
        segments.push(current);
    }

    segments
}

/// Does this token look like a step marker (e.g. "1:", "2.", "10:", "A:" followed by content)?
/// We only treat digit-prefixed tokens as step markers to avoid mis-splitting on "σ:" etc.
fn looks_like_step_marker(token: &str) -> bool {
    let b = token.as_bytes();
    if b.len() < 2 {
        return false;
    }
    // Must start with a digit
    if !b[0].is_ascii_digit() {
        return false;
    }
    // Walk through digits
    let mut i = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    // After the digits we need `:` or `.` and it must be the END of the token
    // (e.g., "1:" or "10.", but NOT "2ℓ" or "3-party").
    i < b.len() && (b[i] == b':' || b[i] == b'.') && i + 1 == b.len()
}

/// If `seg` starts with a step number like "1:" or "1.", return (number, rest).
fn extract_leading_step_number(seg: &str) -> Option<(usize, &str)> {
    let b = seg.as_bytes();
    let mut i = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i >= b.len() {
        return None;
    }
    if b[i] != b':' && b[i] != b'.' {
        return None;
    }
    let n: usize = seg[..i].parse().ok()?;
    let rest = &seg[i + 1..];
    Some((n, rest))
}

/// If `seg` starts with a known label like "Input:", return (label_without_colon, rest).
fn extract_leading_label(seg: &str) -> Option<(&str, &str)> {
    for label in ALGO_BLOCK_LABELS {
        if let Some(rest) = seg.strip_prefix(label) {
            // Drop the trailing colon from the label for the bold text
            let label_no_colon = label.trim_end_matches(':');
            return Some((label_no_colon, rest));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotes_protocol_heading() {
        let md = "- Protocol 1 Bit2A protocol (Π1) Input: A Boolean sharing. Output: An arithmetic sharing. Setup phase: 1: sample r. 2: compute v.";
        let out = postprocess_algorithm_blocks(md);
        assert!(out.contains("#### Protocol 1 Bit2A protocol (Π1)"));
        assert!(out.contains("**Input**"));
        assert!(out.contains("**Output**"));
        assert!(out.contains("**Setup phase**"));
        assert!(out.contains("1. sample r"));
        assert!(out.contains("2. compute v"));
    }

    #[test]
    fn promotes_functionality_heading() {
        let md = "- Functionality: JbK A ← Bit2A(JbKB, λ) Input: A sharing. Output: An output.";
        let out = postprocess_algorithm_blocks(md);
        assert!(out.contains("#### Functionality: JbK A"));
    }

    #[test]
    fn leaves_normal_bullets_alone() {
        let md = "- This is just a regular bullet point with no keywords.";
        let out = postprocess_algorithm_blocks(md);
        assert_eq!(out.trim(), md.trim());
    }

    #[test]
    fn leaves_headings_alone() {
        let md = "## I. Introduction\n\nSome text here.\n";
        let out = postprocess_algorithm_blocks(md);
        assert!(out.contains("## I. Introduction"));
    }
}
