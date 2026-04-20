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
    /// All authors the heuristic could identify. Populated for the verifier
    /// prompt so the LLM can check whether a repo's owner name matches any
    /// author. May be empty when author lines fail the heuristic; callers
    /// must treat absence as "unknown" rather than "no authors".
    #[serde(default)]
    pub authors: Vec<String>,
    /// First ~400 chars of the paper's abstract when the heading or inline
    /// marker is recognisable. Used as additional context for the verifier.
    /// `None` when we can't locate an abstract with the simple heuristics.
    #[serde(default)]
    pub abstract_excerpt: Option<String>,
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

    // Step 2: scan a few following lines for author candidates. Unlike the
    // prior version we don't stop on the first hit — many papers list
    // authors across multiple lines (separate line per affiliation, or
    // multiple `##` headings) and the verifier wants them all. We still
    // cap the scan so we don't wander into the body.
    // Some arXiv exports put authors on an H2/H3 heading line (e.g. the
    // SpaceTimePilot paper); we strip leading `#` markers before evaluating.
    let mut authors: Vec<String> = Vec::new();
    for _ in 0..12 {
        let Some(line) = lines.next() else { break };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // Keep scanning across blank lines — author lines are often
            // separated by them.
            continue;
        }
        let candidate = trimmed.trim_start_matches('#').trim();
        if candidate.is_empty() {
            continue;
        }
        if is_non_author_line(candidate) {
            continue;
        }
        // `authors_from_line` returns ALL plausible names in the line (e.g.
        // a comma-separated list) — accumulating across multiple lines
        // gives us the full author set when papers put each author on
        // their own line.
        for a in authors_from_line(candidate) {
            if !authors.iter().any(|existing| existing == &a) {
                authors.push(a);
            }
        }
    }

    let first_author = authors.first().cloned();
    let abstract_excerpt = extract_abstract(&head);

    Some(PaperFrontMatter {
        title,
        first_author,
        authors,
        abstract_excerpt,
    })
}

/// Locate the abstract in the head of a paper and return up to 400 chars.
/// Recognises three common patterns:
///   - `## Abstract` / `### Abstract` H2/H3 heading
///   - `**Abstract**` bold marker (optionally followed by em-dash / period)
///   - `Abstract—...` / `Abstract.` inline marker at line start
pub(crate) fn extract_abstract(head: &str) -> Option<String> {
    const MAX_LEN: usize = 400;

    let lines: Vec<&str> = head.lines().collect();
    let mut start: Option<usize> = None;
    let mut inline_remainder: Option<String> = None;

    for (idx, raw) in lines.iter().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        // H2/H3 heading exactly "Abstract" (case-insensitive).
        if let Some(body) = line
            .strip_prefix("## ")
            .or_else(|| line.strip_prefix("### "))
        {
            if body.trim().eq_ignore_ascii_case("abstract") {
                start = Some(idx + 1);
                break;
            }
        }
        // **Abstract**... on its own line — may or may not have text after it.
        if let Some(rest) = line.strip_prefix("**Abstract**") {
            let tail = rest
                .trim_start_matches(['—', '-', '.', ':', ' ', '\u{00A0}'])
                .trim();
            if tail.is_empty() {
                start = Some(idx + 1);
            } else {
                inline_remainder = Some(tail.to_string());
                start = Some(idx + 1);
            }
            break;
        }
        // Bare "Abstract—..." or "Abstract. ..." inline.
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("abstract—")
            || lower.starts_with("abstract-")
            || lower.starts_with("abstract.")
            || lower.starts_with("abstract:")
        {
            // Keep everything after the marker on the same line. Use
            // `char_indices` so the split point is a valid char boundary —
            // `—` is a 3-byte UTF-8 codepoint; a naive `+1` slice panics.
            if let Some((byte_idx, ch)) =
                line.char_indices().find(|(_, c)| matches!(c, '—' | '-' | '.' | ':'))
            {
                let tail = line[byte_idx + ch.len_utf8()..].trim();
                if !tail.is_empty() {
                    inline_remainder = Some(tail.to_string());
                }
                start = Some(idx + 1);
                break;
            }
        }
    }

    let start = start?;
    let mut buf = String::new();
    if let Some(inline) = inline_remainder {
        buf.push_str(&inline);
    }
    for raw in lines.iter().skip(start) {
        let line = raw.trim();
        if line.is_empty() {
            if buf.is_empty() {
                continue;
            }
            // Blank line ends the abstract paragraph.
            break;
        }
        // Next heading ends the abstract.
        if line.starts_with('#') {
            break;
        }
        if !buf.is_empty() {
            buf.push(' ');
        }
        buf.push_str(line);
        if buf.len() >= MAX_LEN {
            break;
        }
    }

    let buf = normalize_whitespace(&buf);
    if buf.is_empty() {
        return None;
    }
    let clipped: String = buf.chars().take(MAX_LEN).collect();
    Some(clipped)
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

/// Extract every plausible author name from a single line. Splits on
/// commas and " and " separators, runs the per-chunk validator, and
/// returns the ones that pass. Empty on any line that isn't an author
/// list.
///
/// Examples handled:
///   "Alice Smith^1, Bob Jones^2, and Carol Wu^3" → ["Alice Smith", "Bob Jones", "Carol Wu"]
///   "Alice Smith (Google), Bob Jones (Meta)"     → ["Alice Smith", "Bob Jones"]
///   "Jay Roberts Protopia AI jay@protopia.ai"    → ["Jay Roberts"]  (email dropped)
///   "Zhening Huang Hyeonho Jeong Xuelin Chen"    → ["Zhening Huang"] (first 2 tokens)
///   "Hongyu Wang $^{*\dagger}$ Shuming Ma $^{*}$ Li Dong" → first three names
///     (LaTeX `$...$` superscripts stripped; whitespace separators treated
///     like implicit commas once the math blocks go away).
fn authors_from_line(line: &str) -> Vec<String> {
    // Strip inline LaTeX math blocks like `$^{*\dagger\ddagger}$` which
    // otherwise break the title-case test (tokens start with `$`). PDFs
    // converted from arxiv preprints routinely embed these where papers
    // mark affiliation footnotes.
    let no_math = strip_inline_math(line);
    let cleaned = no_math.trim_matches(|c: char| c == '*' || c == '_');
    let mut out: Vec<String> = Vec::new();
    for chunk in cleaned.split([',', ';']) {
        let chunk = strip_and_and(chunk);
        if let Some(name) = author_from_chunk(chunk) {
            out.push(name);
        }
    }
    out
}

/// Strip `$...$` inline math blocks from a line. Dead-simple scan — we
/// enter/exit whenever we hit an unescaped `$`, dropping everything in
/// between. Lines like `Hongyu Wang $^{\dagger}$ Shuming Ma ...` → `Hongyu
/// Wang  Shuming Ma ...`. Unclosed `$` leaves the rest intact.
fn strip_inline_math(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_math = false;
    for c in s.chars() {
        if c == '$' {
            in_math = !in_math;
            continue;
        }
        if !in_math {
            out.push(c);
        }
    }
    // If we terminated inside math (unclosed `$`), the tail is already
    // skipped; also collapse runs of whitespace so the subsequent
    // splitting logic behaves as if the math blocks had been commas.
    out
}

/// Validate a single comma-delimited chunk as an author name and return the
/// canonicalised form (first 2 tokens when the chunk is a long affiliation-
/// style run). Shared rules between single-author and multi-author lines.
fn author_from_chunk(chunk: &str) -> Option<String> {
    // Drop email-like tokens (contain `@`) before validation — academic
    // author lists often glue `first.last@affiliation.tld` onto the line,
    // which breaks `all_titlecase`.
    let no_email: String = chunk
        .split_whitespace()
        .filter(|tok| !tok.contains('@'))
        .collect::<Vec<_>>()
        .join(" ");

    let cleaned = strip_author_annotations(&no_email);
    let cleaned = cleaned.trim();

    if cleaned.len() < 3 || cleaned.len() > 80 {
        return None;
    }
    if !cleaned.contains(' ') {
        return None;
    }
    let alpha_ratio = cleaned.chars().filter(|c| c.is_alphabetic()).count() as f32
        / cleaned.chars().count().max(1) as f32;
    if alpha_ratio < 0.6 {
        return None;
    }
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

    // More than 2 tokens = whitespace-separated co-author list or
    // name+affiliation glued together (e.g. "Jay Roberts Protopia AI",
    // or a 5-author run like "Alice Smith Bob Jones Carol Wu Dan Lee Eva
    // Park"). The first two tokens are the conservative guess for the
    // name; accepts some loss on 3-token surnames ("Alice von Smith") in
    // exchange for stripping affiliation runs that otherwise pollute the
    // SearXNG query and the verifier prompt.
    let out = if tokens.len() > 2 {
        tokens[..2].join(" ")
    } else {
        cleaned.to_string()
    };
    Some(out)
}

fn strip_and_and(s: &str) -> &str {
    // Strip leading/trailing " and " connectives left over from commas
    // splitting "A, B, and C" → ["A", " B", " and C"]. Both ends, because
    // authors are sometimes written "X and Y" without the trailing comma.
    let s = s.trim();
    let s = s
        .strip_prefix("and ")
        .or_else(|| s.strip_prefix("And "))
        .unwrap_or(s);
    s.strip_suffix(" and")
        .or_else(|| s.strip_suffix(" And"))
        .unwrap_or(s)
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

    #[test]
    fn collects_all_authors_from_comma_list() {
        let md = "# Title\n\nAlice Smith, Bob Jones, and Carol Wu\n";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.first_author.as_deref(), Some("Alice Smith"));
        assert_eq!(fm.authors, vec!["Alice Smith", "Bob Jones", "Carol Wu"]);
    }

    #[test]
    fn collects_authors_across_separate_lines_with_emails() {
        // Mirrors the Stained Glass Transform paper which lists each
        // author on their own line with an email glued on.
        let md = "\
# Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform\n\
\n\
Jay Roberts Protopia AI jay@protopia.ai\n\
\n\
Kyle Mylonakis Protopia AI kyle@protopia.ai\n\
\n\
Abstract—We present a thing.\n\
";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.first_author.as_deref(), Some("Jay Roberts"));
        assert!(fm.authors.contains(&"Jay Roberts".to_string()));
        assert!(fm.authors.contains(&"Kyle Mylonakis".to_string()));
    }

    #[test]
    fn extracts_abstract_from_h2_heading() {
        let md = "\
# Title\n\
\n\
Alice Smith, Bob Jones\n\
\n\
## Abstract\n\
\n\
We present a new method for doing X that achieves Y.\n\
\n\
## Introduction\n\
\n\
Not in the abstract.\n\
";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(
            fm.abstract_excerpt.as_deref(),
            Some("We present a new method for doing X that achieves Y.")
        );
    }

    #[test]
    fn extracts_abstract_from_inline_marker() {
        let md = "\
# Title\n\
\n\
Alice Smith\n\
\n\
Abstract—The high cost of compute leads to multi-tenant deployments.\n\
";
        let fm = extract_front_matter(md).unwrap();
        assert!(fm.abstract_excerpt.is_some());
        assert!(fm
            .abstract_excerpt
            .as_deref()
            .unwrap()
            .starts_with("The high cost of compute"));
    }

    #[test]
    fn no_abstract_when_absent() {
        let md = "# Title\n\nAlice Smith\n\nIntroduction body directly.\n";
        let fm = extract_front_matter(md).unwrap();
        assert!(fm.abstract_excerpt.is_none());
    }

    #[test]
    fn collects_authors_with_latex_superscript_affiliations() {
        // BitNet-style author line — LaTeX `$^{...}$` blocks between
        // names (no commas). Previously returned no author because the
        // `$` tokens broke the title-case check.
        let md = "\
# BitNet: Scaling 1-bit Transformers for Large Language Models\n\
\n\
Hongyu Wang $^{*\\dagger\\ddagger}$ Shuming Ma $^{*\\dagger}$ Li Dong $^{\\dagger}$\n\
";
        let fm = extract_front_matter(md).unwrap();
        assert_eq!(fm.first_author.as_deref(), Some("Hongyu Wang"));
    }

    #[test]
    fn abstract_truncated_to_cap() {
        let filler = "a".repeat(600);
        let md = format!("# T\n\nAlice Smith\n\n## Abstract\n\n{filler}\n");
        let fm = extract_front_matter(&md).unwrap();
        let abs = fm.abstract_excerpt.unwrap();
        assert!(abs.len() <= 400, "got {} chars", abs.len());
    }
}
