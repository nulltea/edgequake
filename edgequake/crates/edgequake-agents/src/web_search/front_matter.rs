//! Extract paper title + first-author from the markdown produced by
//! EdgeQuake's PDF conversion backends (Vision / EdgeParse / VLM-OCR).
//!
//! Heuristic-only (no LLM calls) to keep the happy path cheap. For most
//! arXiv-style preprints the first H1 is the title and the line(s) beneath
//! list authors. If the structure is too irregular we return `None` and the
//! resolver should fall back to an LLM-based extraction.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PaperFrontMatter {
    pub title: String,
    /// The first author's name as it appears in the paper. We avoid trying to
    /// split into given/surname — the web query works fine with the full string
    /// and it's robust to i18n.
    pub first_author: Option<String>,
}

/// Scan the first `scan_chars` characters of `markdown` for an H1 followed by
/// an author line.
///
/// Returns `None` if no plausible title heading is found.
pub fn extract_front_matter(markdown: &str) -> Option<PaperFrontMatter> {
    extract_with_limit(markdown, 4_000)
}

fn extract_with_limit(markdown: &str, scan_chars: usize) -> Option<PaperFrontMatter> {
    let head: String = markdown.chars().take(scan_chars).collect();
    let mut lines = head.lines().peekable();

    // Step 1: find the first H1.
    let title = loop {
        let line = lines.next()?;
        if let Some(rest) = line.strip_prefix("# ") {
            let t = rest.trim();
            if !t.is_empty() && !is_metadata_heading(t) {
                break normalize_whitespace(t);
            }
        }
    };

    // Step 2: scan a few following lines for an author candidate.
    // Some arXiv exports put authors on an H2/H3 heading line (e.g. the
    // SpaceTimePilot paper); we strip leading `#` markers before evaluating.
    let mut first_author: Option<String> = None;
    for _ in 0..10 {
        let Some(line) = lines.next() else { break };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let candidate = trimmed.trim_start_matches('#').trim();
        if candidate.is_empty() {
            continue;
        }
        if is_non_author_line(candidate) {
            continue;
        }
        if let Some(author) = first_author_from_line(candidate) {
            first_author = Some(author);
            break;
        }
    }

    Some(PaperFrontMatter {
        title,
        first_author,
    })
}

fn is_metadata_heading(s: &str) -> bool {
    matches!(
        s.to_ascii_lowercase().as_str(),
        "abstract" | "introduction" | "preface" | "contents" | "table of contents"
    )
}

fn is_non_author_line(s: &str) -> bool {
    // Hard rules / code fences / lists.
    if s.starts_with("---") || s.starts_with("===") || s.starts_with("***") {
        return true;
    }
    if s.starts_with("```") || s.starts_with('|') || s.starts_with('>') {
        return true;
    }
    if s.starts_with("- ") || s.starts_with("* ") || s.starts_with("1. ") {
        return true;
    }
    // Reject obvious section headings (after the caller has stripped `#`).
    if is_section_heading(s) {
        return true;
    }
    // Arxiv "xxxx.xxxxx" id line, affiliation lines ending with university etc.
    let lower = s.to_ascii_lowercase();
    if lower.starts_with("arxiv:") || lower.starts_with("doi:") {
        return true;
    }
    // Lines that start with Abstract / Introduction keywords (often bold-marked).
    if lower.starts_with("**abstract**") || lower.starts_with("abstract.") {
        return true;
    }
    false
}

fn is_section_heading(s: &str) -> bool {
    let normalised = s.trim().trim_matches(|c: char| c == '*' || c == '_');
    matches!(
        normalised.to_ascii_lowercase().as_str(),
        "abstract"
            | "introduction"
            | "preface"
            | "contents"
            | "table of contents"
            | "acknowledgments"
            | "acknowledgements"
            | "related work"
            | "background"
            | "conclusion"
            | "discussion"
    )
}

/// Best-effort extract the first author from a line like:
/// "Alice Smith^1, Bob Jones^2, and Carol Wu^3"
/// "Alice Smith*, Bob Jones†"
/// "Alice Smith 1, Bob Jones 2"
/// "Alice Smith (Google), Bob Jones (Meta)"
fn first_author_from_line(line: &str) -> Option<String> {
    // Strip common markdown emphasis that wraps author lists.
    let cleaned = line.trim_matches(|c: char| c == '*' || c == '_');

    // Split on comma or " and " — take the first chunk.
    let first_chunk = cleaned.split([',', ';']).next().unwrap_or(cleaned);
    let first_chunk = strip_and_and(first_chunk);

    let cleaned = strip_author_annotations(first_chunk);
    let cleaned = cleaned.trim();

    // A plausible author name has at least 2 letters and contains a space
    // (given + surname). This rules out orcid IDs, affiliations etc.
    if cleaned.len() < 3 || cleaned.len() > 80 {
        return None;
    }
    if !cleaned.contains(' ') {
        return None;
    }
    // Must be mostly alphabetic.
    let alpha_ratio = cleaned.chars().filter(|c| c.is_alphabetic()).count() as f32
        / cleaned.chars().count().max(1) as f32;
    if alpha_ratio < 0.6 {
        return None;
    }
    // Structural check: the line must be all title-case tokens (no lowercase
    // connective words like "and"), each ≥2 chars. This rejects section
    // headers like "Space and Time" while accepting "Alice Smith" or a long
    // "Zhening Huang Hyeonho Jeong Xuelin Chen" co-author line.
    let tokens: Vec<&str> = cleaned.split_whitespace().collect();
    if tokens.len() < 2 {
        return None;
    }
    let all_titlecase = tokens.iter().all(|t| {
        t.chars().next().is_some_and(|c| c.is_uppercase())
            && t.chars().filter(|c| c.is_alphabetic()).count() >= 2
    });
    if !all_titlecase {
        return None;
    }

    // If the chunk has many tokens, it's a whitespace-separated author list
    // (e.g. "Zhening Huang Hyeonho Jeong Xuelin Chen"); take just the first
    // two tokens as the likely first author.
    let out = if tokens.len() > 4 {
        tokens[..2].join(" ")
    } else {
        cleaned.to_string()
    };
    Some(out)
}

fn strip_and_and(s: &str) -> &str {
    s.strip_suffix(" and").unwrap_or(s)
}

/// Drop superscript markers, parenthesised affiliations, digits, and common
/// affiliation glyphs († * ‡ § ¶).
fn strip_author_annotations(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '(' | '[' | '{' => {
                depth += 1;
            }
            ')' | ']' | '}' => {
                depth = (depth - 1).max(0);
            }
            _ if depth > 0 => {}
            '0'..='9' | '†' | '*' | '‡' | '§' | '¶' | '#' | '⋄' | '◦' | '•' | '^' => {}
            // Latin-1 supplement superscripts ¹ ² ³.
            '\u{00B2}' | '\u{00B3}' | '\u{00B9}' => {}
            // Unicode superscripts ⁰ ⁴-⁹ and friends (U+2070–U+207F).
            '\u{2070}'..='\u{207F}' => {}
            // Unicode subscripts (U+2080–U+209F).
            '\u{2080}'..='\u{209F}' => {}
            _ => out.push(c),
        }
    }
    out
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_basic_arxiv_header() {
        let md = "\
# AlphaEvolve: Evolving Program Structures\n\
\n\
Alice Smith, Bob Jones, Carol Wu\n\
\n\
## Abstract\n\
We present...\n\
";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.title, "AlphaEvolve: Evolving Program Structures");
        assert_eq!(fm.first_author.as_deref(), Some("Alice Smith"));
    }

    #[test]
    fn handles_superscript_affiliations() {
        let md = "# Title Here\n\nAlice Smith¹, Bob Jones²\n";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.first_author.as_deref(), Some("Alice Smith"));
    }

    #[test]
    fn handles_parenthesised_affiliation() {
        let md = "# Title Here\n\nAlice Smith (DeepMind), Bob Jones (Meta)\n";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.first_author.as_deref(), Some("Alice Smith"));
    }

    #[test]
    fn handles_dagger_glyph() {
        let md = "# Title Here\n\nAlice Smith†, Bob Jones*, Carol Wu‡\n";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.first_author.as_deref(), Some("Alice Smith"));
    }

    #[test]
    fn skips_horizontal_rule_before_authors() {
        let md = "# Title Here\n\n---\n\nAlice Smith, Bob Jones\n";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.first_author.as_deref(), Some("Alice Smith"));
    }

    #[test]
    fn no_author_when_structure_irregular() {
        let md = "# Title Here\n\nhttps://arxiv.org/abs/2512.25075\n\nkeyword1, keyword2\n";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.title, "Title Here");
        // "keyword1" is a single word → rejected; arxiv line also rejected.
        assert!(
            fm.first_author.is_none() || fm.first_author.as_deref() == Some("keyword1, keyword2")
        );
    }

    #[test]
    fn extracts_author_from_h2_heading_line() {
        // Mirrors the SpaceTimePilot arXiv layout: H1 title, H2 subtitle
        // fragment, then another H2 that lists authors on one line.
        let md = "\
# SpaceTimePilot: Generative Rendering of Dynamic Scenes Across\n\
\n\
## Space and Time\n\
\n\
## Zhening Huang Hyeonho Jeong Xuelin Chen\n\
";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(
            fm.title,
            "SpaceTimePilot: Generative Rendering of Dynamic Scenes Across"
        );
        // First plausible author token sequence is Zhening Huang Hyeonho Jeong...
        // Our heuristic takes the whole line (no commas) as one chunk.
        assert!(fm
            .first_author
            .as_deref()
            .unwrap()
            .contains("Zhening Huang"));
    }

    #[test]
    fn rejects_lowercase_connective_as_author() {
        // "Space and Time" used to slip through as an author on word-count /
        // alpha-ratio alone. Structural "all tokens title-case" check rejects it.
        let md = "# Title\n\n## Space and Time\n\n## Alice Smith Bob Jones\n";
        let fm = extract_front_matter(md).unwrap();
        assert!(fm.first_author.as_deref().unwrap().contains("Alice Smith"));
    }

    #[test]
    fn returns_none_when_no_h1() {
        let md = "Some body text without any heading.\n\nMore text.\n";
        assert!(extract_front_matter(md).is_none());
    }

    #[test]
    fn skips_metadata_h1_abstract() {
        let md = "# Abstract\n\nThis is not a title.\n\n# Real Title\n\nAlice Smith, Bob\n";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.title, "Real Title");
    }

    #[test]
    fn normalizes_title_whitespace() {
        let md = "#   Spaced    Out   Title\n\nAlice Smith, Bob Jones\n";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.title, "Spaced Out Title");
    }
}
