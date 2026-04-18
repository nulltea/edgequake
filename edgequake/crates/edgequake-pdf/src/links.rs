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
}
