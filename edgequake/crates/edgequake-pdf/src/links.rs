//! PDF hyperlink extraction.
//!
//! Academic papers embed clickable URLs as PDF `Link` annotations with
//! `URI` actions. Reading them directly — instead of OCR'ing the page — gives
//! us perfect URLs plus their page and y-position, which we use to:
//!
//! 1. Identify the paper's reference implementation (GitHub/GitLab/Bitbucket).
//! 2. Filter out URLs that appear past the References/Bibliography heading,
//!    since those cite *other* papers' repos, not the paper's own.
//!
//! pdfium is not async-safe; callers must invoke [`extract_links`] from a
//! blocking context (e.g. `tokio::task::spawn_blocking`).
//!
//! Companion module [`crate::repos`] filters the raw extraction down to
//! a ranked list of candidate reference repos.

use std::borrow::Cow;

use pdfium_render::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A URI link annotation extracted from a PDF page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfLinkAnnotation {
    pub url: String,
    /// Zero-based page index.
    pub page_index: usize,
    /// y-coordinate of the link's top edge, in PDF points.
    /// Pdfium uses a bottom-left origin, so larger values = higher on the page.
    pub y_top: f32,
}

/// Location of the "References" / "Bibliography" / "Works Cited" heading.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReferenceBoundary {
    pub page_index: usize,
    /// y-coordinate of the heading's top edge (PDF points, bottom-left origin).
    pub y_top: f32,
}

/// Result of scanning a PDF for link annotations and the references-section boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkExtraction {
    pub links: Vec<PdfLinkAnnotation>,
    pub refs_boundary: Option<ReferenceBoundary>,
    pub page_count: usize,
}

impl LinkExtraction {
    /// True when `link` sits strictly below (= later in reading order than)
    /// the references-section heading.
    pub fn is_past_refs(&self, link: &PdfLinkAnnotation) -> bool {
        let Some(b) = self.refs_boundary else {
            return false;
        };
        match link.page_index.cmp(&b.page_index) {
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Greater => true,
            // Same page: pdfium y grows upward, so "below the heading" means
            // y_top < heading.y_top.
            std::cmp::Ordering::Equal => link.y_top < b.y_top,
        }
    }
}

#[derive(Debug, Error)]
pub enum PdfLinkError {
    #[error("pdfium failed to load library: {0}")]
    LibraryLoad(String),
    #[error("pdfium failed to parse PDF: {0}")]
    Parse(String),
}

/// Load pdfium using the same bundled/auto strategy as `edgequake-pdf2md`.
fn get_pdfium() -> Result<Pdfium, PdfLinkError> {
    #[cfg(feature = "bundled")]
    {
        pdfium_auto::bind_bundled().map_err(|e| PdfLinkError::LibraryLoad(e.to_string()))
    }
    #[cfg(not(feature = "bundled"))]
    pdfium_auto::bind_pdfium_silent().map_err(|e| PdfLinkError::LibraryLoad(e.to_string()))
}

/// Regex matching a line that is *only* a references-section heading.
///
/// We accept `References`, `Bibliography`, `Works Cited`, with optional
/// numbering (`6.`, `6 `) and in any case. Required anchors prevent
/// false-positives like "References and notes" mid-paragraph.
fn refs_heading_regex() -> &'static regex::Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::RegexBuilder::new(
            r"^\s*(?:\d+\.?\s+)?(?:references|bibliography|works\s+cited|literature\s+cited)\s*$",
        )
        .case_insensitive(true)
        .build()
        .expect("static regex compiles")
    })
}

/// Regex matching the start of a *non-references* section that follows the
/// bibliography — used to detect where the References section ends so we
/// can preserve the appendix (which usually follows References in academic
/// papers). Two arms:
///
/// 1. `Appendix` / `Supplementary` / `Supplement` at the start of the
///    line. First letter must be capital `A`/`S` (so lowercase
///    `appendix` mid-paragraph doesn't false-match); the rest is
///    case-insensitive (`Appendix`, `APPENDIX`, `Supplement`,
///    `SUPPLEMENTARY`).
/// 2. A letter-headed section: `A Notation`, `A. Notation`, `A) Proofs`,
///    `D.1 IMA inverter`, `D.1. IMA inverter`. The trailing body is
///    restricted to letters/digits/hyphens/spaces (no commas, no
///    parentheses) so that bibliography author-list entries like
///    `A. Smith, J. Doe (2024). Title.` cannot match.
//
// Note: the pattern is kept on one line. Splitting it into a `(?x)`
// free-spacing form mis-compiles with the current `regex` crate version
// (escape sequences interact oddly with the alternation across lines and
// the `\.\d+\.?` arm silently fails to match).
fn non_refs_section_heading_regex() -> &'static regex::Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::RegexBuilder::new(
            r"^[AS](?i:ppendix|upplement(?:ary)?)\b|^[A-Z](?:\.\d+\.?|[.)])?\s+[A-Z][A-Za-z\d\- ]{0,120}$",
        )
        .build()
        .expect("static regex compiles")
    })
}

/// True when `line` opens a new (non-references) section after the
/// bibliography. We treat any markdown heading at the same or higher
/// importance level (i.e. with the same or fewer `#` markers) as the
/// References heading as a section break — unless the heading itself is
/// another `References` heading (rare duplicated heading from VLM-OCR),
/// in which case we keep scanning. Bare-heading lines fall through to
/// [`non_refs_section_heading_regex`].
fn is_section_start_after_refs(line: &str, refs_hash_level: usize) -> bool {
    let trimmed_left = line.trim_start();
    let hash_count = trimmed_left.chars().take_while(|&c| c == '#').count();
    let cleaned = trimmed_left.trim_start_matches('#').trim();
    if cleaned.is_empty() {
        return false;
    }
    if refs_heading_regex().is_match(cleaned) {
        return false; // Duplicate references heading — keep scanning.
    }
    if hash_count > 0 {
        // Markdown heading. Treat as section break when it is at the same
        // or higher importance than the references heading. When the refs
        // heading itself was bare (level 0), any `#`-marked heading after
        // it must be a new section.
        return refs_hash_level == 0 || hash_count <= refs_hash_level;
    }
    non_refs_section_heading_regex().is_match(cleaned)
}

/// Strip the References / Bibliography section from `markdown` while
/// preserving any sections that follow it (Appendix, Supplementary, etc.).
///
/// Why this exists: the bibliography of an academic paper is dense with
/// author names, journal titles, and URLs that all behave as noise during
/// entity extraction — they get harvested as entities/relationships and
/// pollute the knowledge graph + vector store with citation-only matches.
/// Excluding the bibliography at chunking time keeps it out of every
/// downstream consumer (chunks, embeddings, entities, retrieval) while the
/// full markdown stays available for human viewing in the document Content
/// tab (which renders from `pdf_documents.markdown_content`, not from
/// chunks).
///
/// However, academic papers almost always place the Appendix (notation
/// tables, proofs, additional experiments, reproducibility details) *after*
/// References. The previous version of this function truncated at the
/// first References heading and silently dropped the appendix, making
/// appendix-only content unretrievable. We now find both the start of the
/// References section and the start of the next non-references section
/// (Appendix / Supplementary / `A Notation`-style heading) and splice out
/// just the References block.
///
/// When no References heading is found, returns the input verbatim. When
/// References is found but no subsequent section is detected, falls back
/// to truncate-at-References (the conservative pre-fix behaviour).
///
/// Markdown heading markers (`#`, `##`, …) are tolerated and the matching
/// is anchored to whole lines, so `References and notes` mid-paragraph
/// won't trip the strip.
pub fn strip_references_section(markdown: &str) -> Cow<'_, str> {
    let re = refs_heading_regex();

    // Locate the first line that is purely a references heading.
    let mut cursor: usize = 0;
    let (refs_start, refs_hash_level) = loop {
        if cursor >= markdown.len() {
            return Cow::Borrowed(markdown);
        }
        let line_end = markdown[cursor..]
            .find('\n')
            .map(|p| cursor + p)
            .unwrap_or(markdown.len());
        let line = &markdown[cursor..line_end];
        let trimmed_left = line.trim_start();
        let hash_count = trimmed_left.chars().take_while(|&c| c == '#').count();
        let cleaned = trimmed_left.trim_start_matches('#').trim();
        if re.is_match(cleaned) {
            break (cursor, hash_count);
        }
        // Slicing at line_end is safe — '\n' is single-byte ASCII so
        // line_end is always a UTF-8 boundary. line_end + 1 lands on the
        // start of the next line (or one past the end on the final line
        // without a trailing newline, harmlessly terminating the loop).
        cursor = line_end + 1;
    };

    // Skip past the references-heading line itself, then scan forward for
    // the next section that should survive the strip.
    let after_refs_line = markdown[refs_start..]
        .find('\n')
        .map(|p| refs_start + p + 1)
        .unwrap_or(markdown.len());

    let mut scan = after_refs_line;
    while scan < markdown.len() {
        let line_end = markdown[scan..]
            .find('\n')
            .map(|p| scan + p)
            .unwrap_or(markdown.len());
        let line = &markdown[scan..line_end];
        if is_section_start_after_refs(line, refs_hash_level) {
            // Splice: keep prefix, drop References block, keep suffix.
            let mut out = String::with_capacity(refs_start + (markdown.len() - scan));
            out.push_str(&markdown[..refs_start]);
            out.push_str(&markdown[scan..]);
            return Cow::Owned(out);
        }
        scan = line_end + 1;
    }

    // Bibliography runs to EOF — fall back to truncating.
    Cow::Borrowed(&markdown[..refs_start])
}

/// Extract all URI link annotations from `pdf_bytes` and locate the
/// references-section heading, if present.
///
/// **Blocking**: wrap in `tokio::task::spawn_blocking` when called from async code.
pub fn extract_links(pdf_bytes: &[u8]) -> Result<LinkExtraction, PdfLinkError> {
    let pdfium = get_pdfium()?;
    let document = pdfium
        .load_pdf_from_byte_slice(pdf_bytes, None)
        .map_err(|e| PdfLinkError::Parse(e.to_string()))?;

    let mut links = Vec::new();
    let mut refs_boundary: Option<ReferenceBoundary> = None;
    let re = refs_heading_regex();

    let pages = document.pages();
    let page_count = pages.len() as usize;

    for (page_idx, page) in pages.iter().enumerate() {
        for link in page.links().iter() {
            let Some(action) = link.action() else {
                continue;
            };
            let PdfAction::Uri(uri_action) = action else {
                continue;
            };
            let Ok(url) = uri_action.uri() else { continue };
            let Ok(rect) = link.rect() else { continue };
            links.push(PdfLinkAnnotation {
                url,
                page_index: page_idx,
                y_top: rect.top().value,
            });
        }

        if refs_boundary.is_none() {
            if let Ok(text) = page.text() {
                for segment in text.segments().iter() {
                    let line = segment.text();
                    if re.is_match(line.trim()) {
                        refs_boundary = Some(ReferenceBoundary {
                            page_index: page_idx,
                            y_top: segment.bounds().top().value,
                        });
                        break;
                    }
                }
            }
        }
    }

    Ok(LinkExtraction {
        links,
        refs_boundary,
        page_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_past_refs_handles_all_cases() {
        let ext = LinkExtraction {
            links: vec![],
            refs_boundary: Some(ReferenceBoundary {
                page_index: 5,
                y_top: 400.0,
            }),
            page_count: 10,
        };

        let before_page = PdfLinkAnnotation {
            url: "u".into(),
            page_index: 2,
            y_top: 100.0,
        };
        let after_page = PdfLinkAnnotation {
            url: "u".into(),
            page_index: 6,
            y_top: 700.0,
        };
        let same_page_above = PdfLinkAnnotation {
            url: "u".into(),
            page_index: 5,
            y_top: 500.0,
        };
        let same_page_below = PdfLinkAnnotation {
            url: "u".into(),
            page_index: 5,
            y_top: 300.0,
        };

        assert!(!ext.is_past_refs(&before_page));
        assert!(ext.is_past_refs(&after_page));
        assert!(!ext.is_past_refs(&same_page_above));
        assert!(ext.is_past_refs(&same_page_below));
    }

    #[test]
    fn is_past_refs_returns_false_without_boundary() {
        let ext = LinkExtraction {
            links: vec![],
            refs_boundary: None,
            page_count: 10,
        };
        let link = PdfLinkAnnotation {
            url: "u".into(),
            page_index: 0,
            y_top: 0.0,
        };
        assert!(!ext.is_past_refs(&link));
    }

    #[test]
    fn refs_heading_regex_matches_common_forms() {
        let re = refs_heading_regex();
        for s in [
            "References",
            "REFERENCES",
            "references",
            "Bibliography",
            "BIBLIOGRAPHY",
            "Works Cited",
            "works cited",
            "Literature Cited",
            "6. References",
            "6 References",
            "  References  ",
        ] {
            assert!(re.is_match(s.trim()), "should match: {s:?}");
        }
    }

    #[test]
    fn refs_heading_regex_rejects_false_positives() {
        let re = refs_heading_regex();
        for s in [
            "References and notes",
            "See References",
            "References:",
            "Other references",
            "[10] References",
        ] {
            assert!(!re.is_match(s.trim()), "should NOT match: {s:?}");
        }
    }

    #[test]
    fn strip_references_truncates_at_markdown_heading() {
        let md = "# Title\n\nIntro paragraph.\n\n## Method\n\nWe do X.\n\n## References\n\n[1] Alice et al, 2024.\n[2] Bob, 2025.\n";
        let out = strip_references_section(md);
        assert!(out.ends_with("We do X.\n\n"), "out: {out:?}");
        assert!(!out.contains("References"));
        assert!(!out.contains("Alice"));
    }

    #[test]
    fn strip_references_handles_bibliography_and_works_cited() {
        for heading in ["## Bibliography", "## Works Cited", "## Literature Cited"] {
            let md = format!("Body.\n\n{heading}\n\nCitation 1.\n");
            let out = strip_references_section(&md);
            assert_eq!(out, "Body.\n\n", "heading={heading:?}");
        }
    }

    #[test]
    fn strip_references_returns_input_when_no_heading() {
        let md = "Just a body, no references heading anywhere.\n";
        assert_eq!(strip_references_section(md), md);
    }

    #[test]
    fn strip_references_ignores_mid_paragraph_mentions() {
        let md = "We see References [1] and the references show that…\nMore body.\n";
        assert_eq!(strip_references_section(md), md);
    }

    #[test]
    fn strip_references_handles_numbered_heading() {
        let md = "Body.\n\n6. References\n\n[1] Citation.\n";
        let out = strip_references_section(md);
        assert_eq!(out, "Body.\n\n");
    }

    #[test]
    fn strip_references_handles_bare_heading_without_hash() {
        // Some VLM-OCR output emits the section heading without the `#`s.
        let md = "Body line.\n\nReferences\n\n[1] Citation.\n";
        let out = strip_references_section(md);
        assert_eq!(out, "Body line.\n\n");
    }

    #[test]
    fn strip_references_drops_at_first_heading_only() {
        // If "References" appears twice (rare — duplicated heading in
        // converted PDF) and nothing else follows, strip to EOF. A
        // duplicated `References` heading does not count as the next
        // section; we keep scanning.
        let md = "Body.\n\n## References\n[1] A\n## References\n[2] B\n";
        let out = strip_references_section(md);
        assert_eq!(out, "Body.\n\n");
    }

    #[test]
    fn strip_references_preserves_appendix_with_markdown_heading() {
        // The canonical case: a paper with References between §7 and the
        // Appendix. We must keep the appendix in chunks/embeddings.
        let md = "## Conclusion\n\nWe conclude X.\n\n## References\n\n[1] Alice, A. (2024). Title.\n[2] Bob, B. (2025). Other.\n\n## Appendix A: Notation\n\nLet $f$ denote the inversion model.\n";
        let out = strip_references_section(&md);
        assert!(out.contains("Conclusion"), "out: {out:?}");
        assert!(out.contains("## Appendix A"), "appendix should survive");
        assert!(out.contains("inversion model"), "appendix body should survive");
        assert!(!out.contains("Alice"), "bibliography must be gone");
        assert!(!out.contains("Bob, B."), "bibliography must be gone");
        // The References heading itself is gone.
        assert!(!out.contains("## References"));
    }

    #[test]
    fn strip_references_preserves_letter_appendix_when_refs_is_bare() {
        // Bare-heading variant from VLM-OCR output: no `#` markers anywhere.
        // Appendix uses single-letter prefix ("A Notation").
        let md = "Body.\n\nReferences\n\n[1] Alice et al, 2024.\n\nA Notation\n\nLet $f$ denote the inverter.\n";
        let out = strip_references_section(&md);
        assert!(out.contains("Body."));
        assert!(out.contains("A Notation"));
        assert!(out.contains("inverter"));
        assert!(!out.contains("Alice"));
    }

    #[test]
    fn strip_references_preserves_dotted_letter_appendix() {
        let md = "## References\n[1] X.\n\nD.1 IMA inverter architecture\n\nWe use a 2-layer MLP.\n";
        let out = strip_references_section(&md);
        assert!(out.contains("D.1 IMA inverter architecture"), "out: {out:?}");
        assert!(out.contains("2-layer MLP"));
        assert!(!out.contains("[1] X."));
    }

    #[test]
    fn strip_references_preserves_appendix_when_refs_is_numbered() {
        let md = "Body.\n\n6. References\n\n[1] Alice.\n[2] Bob.\n\nAppendix\n\nDetails.\n";
        let out = strip_references_section(&md);
        assert!(out.contains("Body."));
        assert!(out.contains("Appendix"));
        assert!(out.contains("Details."));
        assert!(!out.contains("Alice"));
    }

    #[test]
    fn strip_references_treats_sub_headings_inside_refs_as_part_of_refs() {
        // `### Primary Sources` inside the References block (level 3) is
        // deeper than `## References` (level 2) → should NOT count as a
        // new section. The actual appendix is at the same level (`##`).
        let md = "## References\n\n### Primary Sources\n\n[1] Alice.\n\n### Secondary Sources\n\n[2] Bob.\n\n## Appendix A\n\nAppendix prose.\n";
        let out = strip_references_section(&md);
        assert!(out.contains("## Appendix A"));
        assert!(out.contains("Appendix prose"));
        assert!(!out.contains("Primary Sources"));
        assert!(!out.contains("Alice"));
    }

    #[test]
    fn strip_references_does_not_pickup_bibliography_entry_as_section_start() {
        // `A. Smith, J. Doe (2024). Title.` looks letter-prefixed but has
        // commas + parens — must NOT trigger the section-start detector.
        let md = "## References\n\nA. Smith, J. Doe (2024). Title of work.\nB. Lee (2023). Another paper.\n";
        let out = strip_references_section(&md);
        // No surviving section → strip to EOF.
        assert_eq!(out, "");
    }
}
