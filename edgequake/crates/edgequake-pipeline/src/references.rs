//! References-section parsing for academic documents.
//!
//! Pure, deterministic, no LLM. Locates the references / bibliography
//! section in a markdown document, splits it into entries, and extracts
//! best-effort DOI / URL fields. Also provides an inline citation-marker
//! scanner (`[n]`, `[n,m,...]`) used by retrieval-time chunk enrichment.
//!
//! Entry splitting is dual-mode:
//!   * **Numbered** lists (`[n]` / `n.` / `n)`) keep their real numbers, so
//!     inline `[n]` markers in chunks resolve to them during enrichment.
//!   * **Author-year** lists (no leading numbers — e.g. ICLR/NeurIPS style)
//!     are split on blank-line paragraph boundaries and assigned sequential
//!     ordinals for display ordering. Their inline citations are
//!     `(Author, Year)`, so there is nothing for the `[n]` marker enrichment
//!     to map — correctly a no-op there, but they still appear in the
//!     Citations tab.
//!
//! A document with no detectable reference section yields an empty result
//! (clean no-op), never an error.

use std::sync::LazyLock;

use regex::Regex;

/// One parsed reference entry.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ParsedReference {
    /// Reference number taken from the list marker (`[n]`, `n.`, or `n)`).
    pub number: u32,
    /// Full entry text — continuation lines folded in, whitespace collapsed.
    pub raw_text: String,
    /// DOI if one appears in the entry (bare `10.xxxx/...` form).
    pub doi: Option<String>,
    /// First `http(s)` URL appearing in the entry, if any.
    pub url: Option<String>,
}

/// Matches a references-section *heading* line. Deliberately stricter than a
/// substring search: the line must be essentially just the keyword (with an
/// optional ATX `#` prefix and/or a leading section number like `6` / `6.`),
/// so prose such as "References to prior work…" does not trip it.
static HEADING_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?im)^\s*(#{0,6})\s*(?:\d+\.?\s+)?(references|bibliography|works\s+cited)\s*:?\s*$")
        .expect("valid heading regex")
});

/// Any ATX markdown heading line — used to find where the references section
/// ends. Capture group 1 is the run of `#` so we can compare heading levels.
static ATX_HEADING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^(#{1,6})\s+\S").expect("valid atx regex"));

/// Start of a numbered reference entry: `[n]`, `n.`, or `n)` at line start.
static ENTRY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s+").expect("valid entry regex"));

/// A standard DOI. Trailing sentence punctuation is trimmed after matching.
static DOI_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+").expect("valid doi regex")
});

/// First `http(s)` URL in an entry.
static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://[^\s)\]]+").expect("valid url regex"));

/// A 4-digit publication year (1900–2099) — used to recognise author-year
/// bibliography entries.
static YEAR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(?:19|20)\d{2}\b").expect("valid year regex"));

/// Inline citation markers: a bracket containing only digits, commas and
/// whitespace (`[5]`, `[2,5,7]`, `[2, 5]`). Ranges (`[2-7]`) are intentionally
/// excluded — the `-` makes the content fail this character class.
static MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(\d+(?:\s*,\s*\d+)*)\]").expect("valid marker regex"));

/// Parse the references section out of a markdown document.
///
/// Returns the parsed entries in document order. Empty when no numbered
/// references heading/section is found.
pub fn parse_references(markdown: &str) -> Vec<ParsedReference> {
    let Some((section, heading_level)) = locate_section(markdown) else {
        return Vec::new();
    };
    split_entries(section, heading_level)
}

/// Locate the references section body (text *after* the heading line) and the
/// `#` level of the heading (0 when the heading carried no `#`).
///
/// Uses the **last** matching heading (skips in-text mentions), and ends the
/// section at the next heading of the *same or higher* level (fewer-or-equal
/// `#`), or EOF.
fn locate_section(markdown: &str) -> Option<(&str, usize)> {
    let heading = HEADING_RE.find_iter(markdown).last()?;
    let level = HEADING_RE
        .captures(heading.as_str())
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().len())
        .unwrap_or(0);

    let body_start = heading.end();
    let body = &markdown[body_start..];

    // A subsequent heading ends the section only if it is the same or a
    // higher level. A plain-text heading (level 0) is treated as level 1 for
    // this comparison, so only top-level `#` headings can close it.
    let threshold = if level == 0 { 1 } else { level };
    let mut end = body.len();
    for caps in ATX_HEADING_RE.captures_iter(body) {
        let hashes = caps.get(1).map(|m| m.as_str().len()).unwrap_or(0);
        if hashes <= threshold {
            end = caps.get(0).unwrap().start();
            break;
        }
    }
    Some((&body[..end], level))
}

/// Minimum number of `[n]`/`n.`/`n)` line-start markers required to treat a
/// section as a *numbered* list. A stray single match (e.g. an enumerated
/// item inside one author-year entry) must not flip the whole section.
const MIN_NUMBERED_MARKERS: usize = 2;

/// Minimum collapsed length for an author-year paragraph to count as an entry
/// — filters stray short lines (page artifacts, lone headings).
const MIN_UNNUMBERED_ENTRY_LEN: usize = 20;

/// Split a section body into reference entries (numbered or author-year).
fn split_entries(section: &str, _heading_level: usize) -> Vec<ParsedReference> {
    let numbered_markers = section
        .lines()
        .filter(|l| ENTRY_RE.is_match(l))
        .count();
    if numbered_markers >= MIN_NUMBERED_MARKERS {
        split_numbered(section)
    } else {
        split_author_year(section)
    }
}

fn make_entry(number: u32, text: &str) -> Option<ParsedReference> {
    let raw_text = collapse_ws(text);
    if raw_text.is_empty() {
        return None;
    }
    Some(ParsedReference {
        number,
        doi: extract_doi(&raw_text),
        url: extract_url(&raw_text),
        raw_text,
    })
}

/// Numbered lists: a new entry starts at each `[n]` / `n.` / `n)` marker;
/// continuation lines fold into the current entry. The captured digit is the
/// reference number, so inline `[n]` markers resolve to it.
fn split_numbered(section: &str) -> Vec<ParsedReference> {
    let mut out: Vec<ParsedReference> = Vec::new();
    let mut current_number: Option<u32> = None;
    let mut current_text = String::new();

    for line in section.lines() {
        if let Some(caps) = ENTRY_RE.captures(line) {
            if let Some(n) = current_number {
                out.extend(make_entry(n, &current_text));
            }
            current_number = caps
                .get(1)
                .or_else(|| caps.get(2))
                .and_then(|m| m.as_str().parse::<u32>().ok());
            current_text.clear();
            current_text.push_str(&line[caps.get(0).unwrap().end()..]);
        } else if current_number.is_some() {
            current_text.push(' ');
            current_text.push_str(line);
        }
        // Lines before the first marker are ignored.
    }
    if let Some(n) = current_number {
        out.extend(make_entry(n, &current_text));
    }
    out
}

/// Author-year lists: no leading numbers, entries separated by blank lines.
/// Each qualifying paragraph becomes one entry with a sequential ordinal.
/// Paragraphs are kept only if they look like a citation (length threshold +
/// a 4-digit year or a DOI/URL), which filters figure/table sentinels and
/// stray lines that can appear inside the section.
fn split_author_year(section: &str) -> Vec<ParsedReference> {
    let mut out: Vec<ParsedReference> = Vec::new();
    let mut buf = String::new();
    let mut next_number: u32 = 1;

    let mut flush = |buf: &mut String, out: &mut Vec<ParsedReference>, next: &mut u32| {
        let raw_text = collapse_ws(buf);
        buf.clear();
        if looks_like_reference(&raw_text) {
            if let Some(entry) = make_entry(*next, &raw_text) {
                out.push(entry);
                *next += 1;
            }
        }
    };

    for line in section.lines() {
        if line.trim().is_empty() {
            flush(&mut buf, &mut out, &mut next_number);
        } else {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(line);
        }
    }
    flush(&mut buf, &mut out, &mut next_number);
    out
}

/// Heuristic: does a collapsed paragraph look like a bibliography entry?
fn looks_like_reference(s: &str) -> bool {
    if s.chars().count() < MIN_UNNUMBERED_ENTRY_LEN {
        return false;
    }
    // Skip markdown image / table-figure sentinels and HTML blocks.
    if s.starts_with('!') || s.starts_with('<') || s.contains("](edgequake-") {
        return false;
    }
    YEAR_RE.is_match(s) || DOI_RE.is_match(s) || URL_RE.is_match(s) || s.contains("arXiv")
}

/// Collapse all runs of whitespace (incl. folded newlines) to single spaces.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn extract_doi(s: &str) -> Option<String> {
    DOI_RE
        .find(s)
        .map(|m| m.as_str().trim_end_matches(['.', ',', ';', ')']).to_string())
}

fn extract_url(s: &str) -> Option<String> {
    URL_RE
        .find(s)
        .map(|m| m.as_str().trim_end_matches(['.', ',', ';']).to_string())
}

/// Scan chunk text for inline numbered citation markers and return the cited
/// reference numbers, de-duplicated in first-seen order.
///
/// Recognizes `[5]` and `[2,5,7]` / `[2, 5]`. Does not expand ranges.
pub fn scan_citation_markers(text: &str) -> Vec<u32> {
    let mut seen = Vec::new();
    for caps in MARKER_RE.captures_iter(text) {
        let body = caps.get(1).unwrap().as_str();
        for part in body.split(',') {
            if let Ok(n) = part.trim().parse::<u32>() {
                if !seen.contains(&n) {
                    seen.push(n);
                }
            }
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bracket_numbered_references() {
        let md = "\
# Introduction
Some text citing [1] and [2].

## References
[1] A. Smith, \"A great paper\", Journal, 2020. https://doi.org/10.1234/abcd.5678
[2] B. Jones et al., \"Another paper\",
    Conference, 2021.
";
        let refs = parse_references(md);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].number, 1);
        assert!(refs[0].raw_text.contains("A great paper"));
        assert_eq!(refs[0].doi.as_deref(), Some("10.1234/abcd.5678"));
        assert_eq!(refs[0].url.as_deref(), Some("https://doi.org/10.1234/abcd.5678"));
        // Folded continuation line.
        assert_eq!(refs[1].number, 2);
        assert!(refs[1].raw_text.contains("Another paper\", Conference, 2021."));
    }

    #[test]
    fn parses_dot_and_paren_numbered_references() {
        let md = "\
## Bibliography
1. First entry.
2) Second entry.
";
        let refs = parse_references(md);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].number, 1);
        assert_eq!(refs[1].number, 2);
    }

    #[test]
    fn uses_last_heading_and_section_boundary() {
        let md = "\
# References to prior work are common
This sentence mentions references but is not a heading.

## References
[1] First real entry, a paper title, 2020.
[2] Second real entry, another paper, 2021.

## Appendix
[1] Not a reference — this is an appendix list item.
";
        let refs = parse_references(md);
        assert_eq!(refs.len(), 2);
        assert!(refs[0].raw_text.contains("First real entry"));
        assert!(!refs.iter().any(|r| r.raw_text.contains("Not a reference")));
    }

    #[test]
    fn parses_author_year_unnumbered_references() {
        // ICLR/NeurIPS-style: no leading numbers, blank-line-separated entries.
        let md = "\
## Introduction
Citing (Abadi et al., 2016) inline.

## REFERENCES

Martin Abadi, Andy Chu, Ian Goodfellow. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC, pp. 308-318, 2016.

Yoshua Bengio, Aaron Courville. Representation learning: a review. IEEE TPAMI, 35(8):1798-1828, 2013. doi:10.1109/TPAMI.2013.50

## Appendix
Not a reference.
";
        let refs = parse_references(md);
        assert_eq!(refs.len(), 2);
        // Sequential ordinals assigned in document order.
        assert_eq!(refs[0].number, 1);
        assert!(refs[0].raw_text.contains("Deep learning with differential privacy"));
        assert_eq!(refs[1].number, 2);
        assert_eq!(refs[1].doi.as_deref(), Some("10.1109/TPAMI.2013.50"));
        // Appendix line must not leak in as a reference.
        assert!(!refs.iter().any(|r| r.raw_text.contains("Not a reference")));
    }

    #[test]
    fn author_year_filters_non_reference_paragraphs() {
        let md = "\
## References

Short line.

Jane Q. Researcher, A proper citation with enough length and a year, Journal of Things, 2021.

![tbl_1](edgequake-table)
";
        let refs = parse_references(md);
        assert_eq!(refs.len(), 1);
        assert!(refs[0].raw_text.contains("A proper citation"));
    }

    #[test]
    fn no_references_section_is_noop() {
        let md = "# Title\nBody with (Smith, 2020) author-year citations only.";
        assert!(parse_references(md).is_empty());
    }

    #[test]
    fn scans_single_and_list_markers_no_ranges() {
        let text = "As shown [5], and combined [2, 5, 7] and again [5]. A range [2-7] is ignored.";
        assert_eq!(scan_citation_markers(text), vec![5, 2, 7]);
    }

    #[test]
    fn marker_scan_empty_when_none() {
        assert!(scan_citation_markers("no citations here").is_empty());
    }
}
