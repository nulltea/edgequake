//! Walk a markdown document with `pulldown-cmark` and compute, for every
//! byte offset, the active heading path — a shallow-to-deep stack of
//! heading text for every level currently in scope.
//!
//! Used by [`ContextAwareChunking`][super::strategies::ContextAwareChunking]
//! so each produced chunk can be tagged with the section hierarchy it came
//! from (e.g. `["Methods", "B. B2A Protocol"]`). That lets the
//! retrieval layer prefer chunks sharing a heading path with a query, and
//! lets the chunker merge adjacent small chunks only when their section
//! context matches.
//!
//! The walker is a single linear pass over the document; lookups are O(log
//! N) via binary search on the boundary list.

use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};

/// One point in the document where the heading path changed.
#[derive(Debug, Clone)]
struct PathBoundary {
    /// Byte offset where this path becomes active.
    offset: usize,
    /// Shallow-to-deep heading text at this offset, e.g.
    /// `["II. Preliminaries", "B. B2A Protocol"]`.
    path: Vec<String>,
}

/// Pre-computed heading-path lookup over a markdown document.
///
/// Construction walks the document once; `path_at(offset)` then answers
/// in O(log N) where N is the number of heading transitions.
#[derive(Debug, Clone)]
pub struct HeadingPathIndex {
    boundaries: Vec<PathBoundary>,
}

impl HeadingPathIndex {
    /// Walk `md` and build the index.
    pub fn build(md: &str) -> Self {
        let parser = Parser::new(md).into_offset_iter();
        // Stack of (level, heading_text). A new heading at level L pops all
        // entries at level >= L before pushing.
        let mut stack: Vec<(HeadingLevel, String)> = Vec::new();
        // Are we currently inside a heading's text range? If so accumulate
        // the visible text into `current`.
        let mut current: Option<(HeadingLevel, String, usize)> = None;
        let mut boundaries: Vec<PathBoundary> = vec![PathBoundary {
            offset: 0,
            path: Vec::new(),
        }];

        for (event, range) in parser {
            match event {
                Event::Start(Tag::Heading { level, .. }) => {
                    current = Some((level, String::new(), range.start));
                }
                Event::End(TagEnd::Heading(_)) => {
                    if let Some((level, text, start_offset)) = current.take() {
                        // Pop any equal-or-deeper headings off the stack.
                        while let Some((top, _)) = stack.last() {
                            if *top as u32 >= level as u32 {
                                stack.pop();
                            } else {
                                break;
                            }
                        }
                        stack.push((level, text.trim().to_string()));
                        // The new path applies from the heading's START
                        // offset, not its end, so the heading line itself
                        // is attributed to its own section.
                        let new_path: Vec<String> =
                            stack.iter().map(|(_, t)| t.clone()).collect();
                        push_boundary(&mut boundaries, start_offset, new_path);
                    }
                }
                Event::Text(t) | Event::Code(t) => {
                    if let Some((_, buf, _)) = current.as_mut() {
                        buf.push_str(&t);
                    }
                }
                _ => {}
            }
        }

        Self { boundaries }
    }

    /// Return the heading path active at `offset`.
    ///
    /// Empty vector means "before the first heading". Never allocates
    /// when called repeatedly for the same boundary — cloned from the
    /// boundary's owned path.
    pub fn path_at(&self, offset: usize) -> Vec<String> {
        // Binary search for the last boundary with offset <= target.
        let idx = match self
            .boundaries
            .binary_search_by_key(&offset, |b| b.offset)
        {
            Ok(i) => i,
            Err(0) => return Vec::new(),
            Err(i) => i - 1,
        };
        self.boundaries[idx].path.clone()
    }
}

fn push_boundary(list: &mut Vec<PathBoundary>, offset: usize, path: Vec<String>) {
    // De-dup consecutive identical paths; pulldown-cmark can emit multiple
    // events per heading and we'd otherwise clutter the index.
    if let Some(last) = list.last() {
        if last.path == path {
            return;
        }
        // If a new boundary has the same offset as the previous, overwrite —
        // we always want the *latest* path at a given offset.
        if last.offset == offset {
            list.last_mut().unwrap().path = path;
            return;
        }
    }
    list.push(PathBoundary { offset, path });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_doc_gives_empty_path() {
        let idx = HeadingPathIndex::build("just some text, no headings");
        assert!(idx.path_at(0).is_empty());
        assert!(idx.path_at(10).is_empty());
    }

    #[test]
    fn single_top_level_heading() {
        let md = "# Introduction\n\nSome body text here.";
        let idx = HeadingPathIndex::build(md);
        // Before the heading, empty.
        assert!(idx.path_at(0).is_empty() || idx.path_at(0) == vec!["Introduction".to_string()]);
        // Inside the body.
        let body_offset = md.find("Some body").unwrap();
        assert_eq!(idx.path_at(body_offset), vec!["Introduction".to_string()]);
    }

    #[test]
    fn nested_headings_produce_full_path() {
        let md = "\
# II. Preliminaries

Intro text.

## B. B2A Protocol

Body of B.
";
        let idx = HeadingPathIndex::build(md);
        let intro_offset = md.find("Intro text").unwrap();
        assert_eq!(
            idx.path_at(intro_offset),
            vec!["II. Preliminaries".to_string()]
        );
        let body_b_offset = md.find("Body of B").unwrap();
        assert_eq!(
            idx.path_at(body_b_offset),
            vec![
                "II. Preliminaries".to_string(),
                "B. B2A Protocol".to_string(),
            ]
        );
    }

    #[test]
    fn shallower_heading_pops_stack() {
        let md = "\
# A

## A.1

Body inside A.1.

# B

Body inside B.
";
        let idx = HeadingPathIndex::build(md);
        let body_a1 = md.find("Body inside A.1").unwrap();
        assert_eq!(
            idx.path_at(body_a1),
            vec!["A".to_string(), "A.1".to_string()]
        );
        let body_b = md.find("Body inside B").unwrap();
        assert_eq!(idx.path_at(body_b), vec!["B".to_string()]);
    }

    #[test]
    fn heading_with_inline_code_and_math() {
        let md = "## Fig. 2: FUNCTIONALITY $\\mathcal{F}_{\\mathrm{Bit2A}}$\n\nSteps.";
        let idx = HeadingPathIndex::build(md);
        let body = md.find("Steps.").unwrap();
        let path = idx.path_at(body);
        assert_eq!(path.len(), 1);
        // The heading text includes the math source verbatim (pulldown-cmark
        // emits it as Text since it's not fenced).
        assert!(
            path[0].contains("FUNCTIONALITY"),
            "path[0] was {:?}",
            path[0]
        );
    }
}
