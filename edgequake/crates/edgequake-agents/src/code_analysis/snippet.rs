//! Extract source snippets from a cloned repo on disk.
//!
//! The `code-analyzer` service reports matches as `(file, start_line,
//! end_line)`. Those files live on the shared docker volume mounted into
//! both services. Edgequake reads the snippet text for persistence.
//!
//! Safety: every `repo_root` we receive comes from the analyzer's response,
//! which this code does NOT trust — we canonicalise both paths and reject
//! anything escaping the repo root.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SnippetError {
    #[error("file not found inside repo: {0}")]
    NotFound(String),
    #[error("path escapes repo root: {0}")]
    PathEscape(String),
    #[error("invalid line range: {start}..{end} in file with {total} lines")]
    BadRange {
        start: usize,
        end: usize,
        total: usize,
    },
    #[error("IO error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: io::Error,
    },
}

/// Maximum snippet length we'll persist. Larger ranges are truncated with
/// an ellipsis marker; the user can still click through to GitHub at the
/// pinned commit for the full function.
const MAX_SNIPPET_CHARS: usize = 8_000;

#[derive(Debug, Clone)]
pub struct ExtractedSnippet {
    pub file_path: String,
    pub start_line: i32,
    pub end_line: i32,
    pub text: String,
    /// True if the snippet was clipped at MAX_SNIPPET_CHARS.
    pub truncated: bool,
}

/// Read lines [start, end] (1-indexed, inclusive) from `repo_root / file`.
pub fn extract(
    repo_root: &Path,
    relative_file: &str,
    start_line: i32,
    end_line: i32,
) -> Result<ExtractedSnippet, SnippetError> {
    if start_line < 1 || end_line < start_line {
        return Err(SnippetError::BadRange {
            start: start_line.max(0) as usize,
            end: end_line.max(0) as usize,
            total: 0,
        });
    }
    let full_path = resolve_safe(repo_root, relative_file)?;
    let content = fs::read_to_string(&full_path).map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound {
            SnippetError::NotFound(relative_file.to_string())
        } else {
            SnippetError::Io {
                path: full_path.to_string_lossy().into_owned(),
                source: e,
            }
        }
    })?;

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start = start_line as usize;
    let end = end_line as usize;
    if start == 0 || start > total || end > total {
        return Err(SnippetError::BadRange { start, end, total });
    }

    let mut out = String::new();
    for line in &lines[start - 1..end] {
        out.push_str(line);
        out.push('\n');
    }
    let truncated = out.len() > MAX_SNIPPET_CHARS;
    if truncated {
        out.truncate(MAX_SNIPPET_CHARS);
        out.push_str("\n…[truncated]…\n");
    }
    Ok(ExtractedSnippet {
        file_path: relative_file.to_string(),
        start_line,
        end_line,
        text: out,
        truncated,
    })
}

/// Resolve `relative_file` against `repo_root`, rejecting traversal.
fn resolve_safe(repo_root: &Path, relative_file: &str) -> Result<PathBuf, SnippetError> {
    if relative_file.is_empty() {
        return Err(SnippetError::PathEscape(relative_file.into()));
    }
    if Path::new(relative_file).is_absolute() {
        return Err(SnippetError::PathEscape(relative_file.into()));
    }
    // Canonicalise both and check prefix — defuses `..` and symlinks.
    let root = repo_root.canonicalize().map_err(|e| SnippetError::Io {
        path: repo_root.to_string_lossy().into_owned(),
        source: e,
    })?;
    let candidate = root.join(relative_file);
    let canonical = candidate.canonicalize().map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound {
            SnippetError::NotFound(relative_file.to_string())
        } else {
            SnippetError::Io {
                path: candidate.to_string_lossy().into_owned(),
                source: e,
            }
        }
    })?;
    if !canonical.starts_with(&root) {
        return Err(SnippetError::PathEscape(
            canonical.to_string_lossy().into_owned(),
        ));
    }
    Ok(canonical)
}

/// Infer a tree-sitter-style language name from a file extension.
///
/// Kept deliberately coarse — the UI uses this only for the syntax-highlighter
/// hint. Unknown extensions return "text".
pub fn language_from_path(relative_file: &str) -> &'static str {
    let ext = Path::new(relative_file)
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("rs") => "rust",
        Some("py") | Some("pyi") => "python",
        Some("c") | Some("h") => "c",
        Some("cc") | Some("cpp") | Some("cxx") | Some("hpp") | Some("hxx") => "cpp",
        Some("go") => "go",
        Some("ts") | Some("tsx") => "typescript",
        Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => "javascript",
        Some("java") => "java",
        Some("kt") | Some("kts") => "kotlin",
        Some("swift") => "swift",
        Some("rb") => "ruby",
        Some("php") => "php",
        Some("scala") => "scala",
        Some("cs") => "csharp",
        Some("ipynb") => "python", // jupyter notebooks are mostly python
        _ => "text",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn seed_file(root: &Path, name: &str, content: &str) {
        let path = root.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn extracts_inclusive_range() {
        let dir = tempdir().unwrap();
        seed_file(dir.path(), "a.py", "one\ntwo\nthree\nfour\nfive\n");
        let s = extract(dir.path(), "a.py", 2, 4).unwrap();
        assert_eq!(s.text, "two\nthree\nfour\n");
        assert_eq!(s.start_line, 2);
        assert_eq!(s.end_line, 4);
        assert!(!s.truncated);
    }

    #[test]
    fn rejects_path_traversal() {
        let dir = tempdir().unwrap();
        seed_file(dir.path(), "a.py", "hi\n");
        let e = extract(dir.path(), "../etc/passwd", 1, 1).unwrap_err();
        match e {
            SnippetError::NotFound(_) | SnippetError::PathEscape(_) => (),
            other => panic!("expected NotFound or PathEscape, got {other:?}"),
        }
    }

    #[test]
    fn rejects_absolute_path() {
        let dir = tempdir().unwrap();
        let e = extract(dir.path(), "/etc/passwd", 1, 1).unwrap_err();
        assert!(matches!(e, SnippetError::PathEscape(_)));
    }

    #[test]
    fn rejects_out_of_range() {
        let dir = tempdir().unwrap();
        seed_file(dir.path(), "a.py", "one\ntwo\n");
        let e = extract(dir.path(), "a.py", 5, 10).unwrap_err();
        assert!(matches!(e, SnippetError::BadRange { .. }));
    }

    #[test]
    fn truncates_oversized() {
        let dir = tempdir().unwrap();
        let big: String = (0..20_000).map(|_| 'x').collect();
        seed_file(dir.path(), "a.py", &format!("{big}\n"));
        let s = extract(dir.path(), "a.py", 1, 1).unwrap();
        assert!(s.truncated);
        assert!(s.text.contains("…[truncated]…"));
    }

    #[test]
    fn infers_languages() {
        assert_eq!(language_from_path("foo/bar.py"), "python");
        assert_eq!(language_from_path("src/main.rs"), "rust");
        assert_eq!(language_from_path("x/y.cpp"), "cpp");
        assert_eq!(language_from_path("unknown.xyz"), "text");
        assert_eq!(language_from_path("no_ext"), "text");
    }
}
