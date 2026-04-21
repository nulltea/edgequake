use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use regex::Regex;
use uuid::Uuid;

use crate::code_analysis::{language_from_path, CodeArtifact};

use super::types::{
    CodebaseChunk, CodebaseEdge, CodebaseFile, CodebaseIndexMode, CodebaseSymbol, IndexBuildOutput,
};

#[derive(Debug, Clone)]
pub struct IndexLimits {
    pub max_files: usize,
    pub max_file_bytes: u64,
    pub max_chunks: usize,
}

impl Default for IndexLimits {
    fn default() -> Self {
        Self {
            max_files: env_usize("EDGEQUAKE_REFERENCE_CODEBASE_MAX_FILES", 25_000),
            max_file_bytes: env_u64("EDGEQUAKE_REFERENCE_CODEBASE_MAX_FILE_BYTES", 1_048_576),
            max_chunks: env_usize("EDGEQUAKE_REFERENCE_CODEBASE_MAX_CHUNKS", 20_000),
        }
    }
}

pub struct ReferenceCodebaseIndexer {
    limits: IndexLimits,
}

impl ReferenceCodebaseIndexer {
    pub fn new(limits: IndexLimits) -> Self {
        Self { limits }
    }

    pub fn build(
        &self,
        repo_root: &Path,
        mode: CodebaseIndexMode,
        approved_artifacts: &[CodeArtifact],
    ) -> Result<IndexBuildOutput, IndexerError> {
        let repo_root = repo_root
            .canonicalize()
            .map_err(|e| IndexerError::Io(repo_root.display().to_string(), e))?;

        let mut files = self.scan_files(&repo_root)?;
        let mut symbols = Vec::new();
        let mut symbol_by_name: BTreeMap<String, Vec<Uuid>> = BTreeMap::new();
        let mut symbols_by_file: BTreeMap<Uuid, Vec<CodebaseSymbol>> = BTreeMap::new();
        // Pending edges from the tree-sitter path — resolved to target
        // symbol ids after every file has been parsed (so cross-file
        // refs can resolve).
        let mut pending_edges: Vec<super::treesitter::PendingEdge> = Vec::new();
        // Regex-path residue: file ids whose language has no tree-sitter
        // support fall through to the old extractor for a minimal
        // defines+calls graph. Stored as ids (not refs) so a later mutable
        // pass over `files` doesn't trip the borrow checker.
        let mut regex_files: Vec<Uuid> = Vec::new();

        for file in files.iter().filter(|f| f.skipped_reason.is_none()) {
            let content = read_repo_file(&repo_root, &file.file_path)?;
            match super::treesitter::parse_file(file, &content) {
                Some(out) => {
                    for sym in &out.symbols {
                        symbol_by_name
                            .entry(sym.name.clone())
                            .or_default()
                            .push(sym.id);
                    }
                    symbols_by_file
                        .entry(file.id)
                        .or_default()
                        .extend(out.symbols.iter().cloned());
                    symbols.extend(out.symbols);
                    pending_edges.extend(out.edges);
                }
                None => {
                    // Regex fallback — extract_symbols + extract_call_names.
                    let extracted = extract_symbols(file, &content);
                    for sym in extracted {
                        symbol_by_name
                            .entry(sym.name.clone())
                            .or_default()
                            .push(sym.id);
                        symbols_by_file
                            .entry(file.id)
                            .or_default()
                            .push(sym.clone());
                        symbols.push(sym);
                    }
                    regex_files.push(file.id);
                }
            }
        }

        // Pass 2: vendored reachability. Scan thirdparty/vendor files now that
        // the main graph knows which symbol names it references. Keep only
        // vendored symbols that are targeted by an unresolved `calls` /
        // `references` edge from main — one hop, no transitive cascade through
        // vendored → vendored. Zero outgoing edges from vendored symbols so the
        // graph treats them as leaf nodes.
        let unresolved_names: BTreeSet<String> = pending_edges
            .iter()
            .filter(|p| matches!(p.edge_type.as_str(), "calls" | "references"))
            .filter_map(|p| {
                let k = target_key(&p.target_name);
                if symbol_by_name.contains_key(k) {
                    None
                } else {
                    Some(k.to_string())
                }
            })
            .collect();

        let mut vendored_kept_files: BTreeSet<Uuid> = BTreeSet::new();
        if !unresolved_names.is_empty() {
            for file in files
                .iter()
                .filter(|f| f.skipped_reason.as_deref() == Some("vendored_not_referenced"))
            {
                let content = read_repo_file(&repo_root, &file.file_path)?;
                let Some(out) = super::treesitter::parse_file(file, &content) else {
                    continue;
                };
                for sym in out.symbols {
                    // `target_key` strips `::` / `.` qualifiers so a call to
                    // `seal::encrypt` matches a SEAL symbol named `encrypt`.
                    // Keep only the first hit per name — matches the edge
                    // resolver's `.first()` semantics, so extra copies would
                    // sit as orphan nodes.
                    let key = target_key(&sym.name).to_string();
                    if !unresolved_names.contains(&key) {
                        continue;
                    }
                    if symbol_by_name.contains_key(&key) {
                        continue;
                    }
                    symbol_by_name.entry(sym.name.clone()).or_default().push(sym.id);
                    symbols_by_file
                        .entry(file.id)
                        .or_default()
                        .push(sym.clone());
                    symbols.push(sym);
                    vendored_kept_files.insert(file.id);
                }
            }
        }
        for file in files.iter_mut() {
            if vendored_kept_files.contains(&file.id) {
                file.skipped_reason = None;
            }
        }

        let mut edges = Vec::new();

        // `defines`: file → symbol for every symbol in the index. Same
        // shape across both extractors.
        for file in files.iter().filter(|f| f.skipped_reason.is_none()) {
            if let Some(file_symbols) = symbols_by_file.get(&file.id) {
                for sym in file_symbols {
                    edges.push(CodebaseEdge {
                        edge_type: "defines".to_string(),
                        source_symbol_id: None,
                        target_symbol_id: Some(sym.id),
                        source_file_id: Some(file.id),
                        target_file_id: None,
                        target_name: None,
                        metadata: serde_json::json!({}),
                    });
                }
            }
        }

        // Tree-sitter-path edges: resolve each PendingEdge's
        // `target_name` against the global symbol-name index.
        for p in &pending_edges {
            let target_symbol_id = symbol_by_name
                .get(target_key(&p.target_name))
                .and_then(|ids| ids.first())
                .copied();
            let source_symbol_id = if p.source_symbol_id == Uuid::nil() {
                None // file-level edge (import, top-level reference)
            } else {
                Some(p.source_symbol_id)
            };
            edges.push(CodebaseEdge {
                edge_type: p.edge_type.clone(),
                source_symbol_id,
                target_symbol_id,
                source_file_id: Some(p.source_file_id),
                target_file_id: None,
                target_name: Some(p.target_name.clone()),
                metadata: serde_json::json!({
                    "resolution": if target_symbol_id.is_some() { "local_symbol" } else { "unresolved_name" }
                }),
            });
        }

        // Regex-fallback `calls` edges for languages without tree-sitter.
        let files_by_id: BTreeMap<Uuid, &CodebaseFile> =
            files.iter().map(|f| (f.id, f)).collect();
        for file_id in &regex_files {
            let Some(file) = files_by_id.get(file_id) else {
                continue;
            };
            if let Some(file_symbols) = symbols_by_file.get(&file.id) {
                let content = read_repo_file(&repo_root, &file.file_path)?;
                for sym in file_symbols {
                    let body = slice_lines(&content, sym.start_line, sym.end_line);
                    for call in extract_call_names(&body, &sym.language) {
                        let target_symbol_id = symbol_by_name
                            .get(&call)
                            .and_then(|ids| ids.first())
                            .copied();
                        edges.push(CodebaseEdge {
                            edge_type: "calls".to_string(),
                            source_symbol_id: Some(sym.id),
                            target_symbol_id,
                            source_file_id: Some(file.id),
                            target_file_id: None,
                            target_name: Some(call),
                            metadata: serde_json::json!({
                                "resolution": if target_symbol_id.is_some() { "local_symbol" } else { "unresolved_name" }
                            }),
                        });
                    }
                }
            }
        }

        let chunks = self.build_chunks(
            &repo_root,
            mode,
            &files,
            &symbols_by_file,
            approved_artifacts,
        )?;
        let mut language_set = BTreeSet::new();
        for file in &files {
            if file.skipped_reason.is_none() && file.language != "text" {
                language_set.insert(file.language.clone());
            }
        }

        // Keep skipped files for accounting, but only successfully indexed
        // files have downstream symbols/chunks/edges.
        files.sort_by(|a, b| a.file_path.cmp(&b.file_path));

        Ok(IndexBuildOutput {
            files,
            symbols,
            edges,
            chunks,
            language_set: language_set.into_iter().collect(),
        })
    }

    fn scan_files(&self, repo_root: &Path) -> Result<Vec<CodebaseFile>, IndexerError> {
        let mut out = Vec::new();
        let mut stack = vec![repo_root.to_path_buf()];
        let ignore = IgnoreRules::load(repo_root);

        while let Some(dir) = stack.pop() {
            for entry in
                fs::read_dir(&dir).map_err(|e| IndexerError::Io(dir.display().to_string(), e))?
            {
                let entry = entry.map_err(|e| IndexerError::Io(dir.display().to_string(), e))?;
                let path = entry.path();
                let rel = relative_path(repo_root, &path)?;
                if should_skip_path(&rel) || ignore.matches(&rel) {
                    continue;
                }
                let ty = entry
                    .file_type()
                    .map_err(|e| IndexerError::Io(path.display().to_string(), e))?;
                if ty.is_dir() {
                    stack.push(path);
                    continue;
                }
                if !ty.is_file() {
                    continue;
                }
                if out.len() >= self.limits.max_files {
                    break;
                }

                let metadata = entry
                    .metadata()
                    .map_err(|e| IndexerError::Io(path.display().to_string(), e))?;
                let language = language_from_path(&rel).to_string();
                let skipped_reason = if language == "text" {
                    Some("unsupported_language".to_string())
                } else if metadata.len() > self.limits.max_file_bytes {
                    Some("file_too_large".to_string())
                } else if is_vendored_path(&rel) {
                    // Held back until the reachability pass decides whether
                    // any of this file's symbols are called from non-vendored
                    // code. Cleared to None if a symbol is kept.
                    Some("vendored_not_referenced".to_string())
                } else {
                    None
                };
                let bytes = fs::read(&path).map_err(|e| IndexerError::Io(rel.clone(), e))?;
                let content = String::from_utf8_lossy(&bytes);
                out.push(CodebaseFile {
                    id: Uuid::new_v4(),
                    file_path: rel,
                    language,
                    checksum: format!("{:x}", md5::compute(&bytes)),
                    line_count: content.lines().count() as i32,
                    byte_count: bytes.len() as i32,
                    skipped_reason,
                });
            }
        }
        Ok(out)
    }

    fn build_chunks(
        &self,
        repo_root: &Path,
        mode: CodebaseIndexMode,
        files: &[CodebaseFile],
        symbols_by_file: &BTreeMap<Uuid, Vec<CodebaseSymbol>>,
        approved_artifacts: &[CodeArtifact],
    ) -> Result<Vec<CodebaseChunk>, IndexerError> {
        let mut chunks = Vec::new();
        let anchors = ArtifactAnchors::new(approved_artifacts);

        for file in files.iter().filter(|f| f.skipped_reason.is_none()) {
            let content = read_repo_file(repo_root, &file.file_path)?;
            let symbols = symbols_by_file.get(&file.id).cloned().unwrap_or_default();
            let file_has_anchor = anchors.file_has_anchor(&file.file_path);

            for sym in &symbols {
                let anchor = anchors.match_symbol(sym);
                if mode == CodebaseIndexMode::AlgorithmFocused
                    && anchor.is_none()
                    && !file_has_anchor
                {
                    continue;
                }
                let text = slice_lines(&content, sym.start_line, sym.end_line);
                if text.trim().is_empty() {
                    continue;
                }
                chunks.push(CodebaseChunk {
                    id: Uuid::new_v4(),
                    file_id: file.id,
                    symbol_id: Some(sym.id),
                    // `code_artifact.algorithm_id` became Option<Uuid> in
                    // migration 047 (algorithm re-extraction can orphan
                    // artifacts to NULL). `.and_then` flattens the
                    // Option<Option<Uuid>> the map would otherwise produce.
                    algorithm_id: anchor.as_ref().and_then(|a| a.algorithm_id),
                    code_artifact_id: anchor.as_ref().map(|a| a.id),
                    chunk_kind: if anchor.is_some() {
                        "algorithm_anchor".to_string()
                    } else {
                        "symbol".to_string()
                    },
                    language: file.language.clone(),
                    file_path: file.file_path.clone(),
                    symbol_name: Some(sym.name.clone()),
                    start_line: sym.start_line,
                    end_line: sym.end_line,
                    token_estimate: estimate_tokens(&text),
                    algorithm_focus: if anchor.is_some() { 1.0 } else { 0.35 },
                    content_hash: format!("{:x}", md5::compute(text.as_bytes())),
                    content: text,
                });
                if chunks.len() >= self.limits.max_chunks {
                    return Ok(chunks);
                }
            }

            if mode == CodebaseIndexMode::Full && symbols.is_empty() {
                let text = bounded_file_text(&content);
                if !text.trim().is_empty() {
                    chunks.push(CodebaseChunk {
                        id: Uuid::new_v4(),
                        file_id: file.id,
                        symbol_id: None,
                        algorithm_id: None,
                        code_artifact_id: None,
                        chunk_kind: "file_overview".to_string(),
                        language: file.language.clone(),
                        file_path: file.file_path.clone(),
                        symbol_name: None,
                        start_line: 1,
                        end_line: text.lines().count().max(1) as i32,
                        token_estimate: estimate_tokens(&text),
                        algorithm_focus: 0.0,
                        content_hash: format!("{:x}", md5::compute(text.as_bytes())),
                        content: text,
                    });
                }
            }
        }

        Ok(chunks)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IndexerError {
    #[error("IO error reading {0}: {1}")]
    Io(String, #[source] std::io::Error),
    #[error("path escapes repo root: {0}")]
    PathEscape(String),
}

fn extract_symbols(file: &CodebaseFile, content: &str) -> Vec<CodebaseSymbol> {
    let patterns = symbol_patterns(&file.language);
    let lines: Vec<&str> = content.lines().collect();
    let mut out = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        for (kind, re) in &patterns {
            if let Some(caps) = re.captures(line) {
                let Some(name) = caps.name("name").map(|m| m.as_str().to_string()) else {
                    continue;
                };
                let start = idx as i32 + 1;
                let end = find_symbol_end(&lines, idx, &file.language);
                out.push(CodebaseSymbol {
                    id: Uuid::new_v4(),
                    file_id: file.id,
                    symbol_kind: (*kind).to_string(),
                    qualified_name: format!("{}::{}", file.file_path, name),
                    name,
                    parent_symbol_id: None,
                    file_path: file.file_path.clone(),
                    language: file.language.clone(),
                    start_line: start,
                    end_line: end,
                    start_byte: 0,
                    end_byte: 0,
                    metadata: serde_json::Value::Object(serde_json::Map::new()),
                });
                break;
            }
        }
    }
    out
}

fn symbol_patterns(language: &str) -> Vec<(&'static str, Regex)> {
    let specs = match language {
        "rust" => vec![
            (
                "function",
                r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "struct",
                r"^\s*(?:pub\s+)?struct\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "enum",
                r"^\s*(?:pub\s+)?enum\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "impl",
                r"^\s*impl(?:<[^>]+>)?\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
            ),
        ],
        "python" => vec![
            (
                "function",
                r"^\s*def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
            ),
            ("class", r"^\s*class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"),
        ],
        "typescript" | "javascript" => vec![
            (
                "function",
                r"^\s*(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)",
            ),
            (
                "class",
                r"^\s*(?:export\s+)?class\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)",
            ),
            (
                "function",
                r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\(",
            ),
        ],
        "go" => vec![
            (
                "function",
                r"^\s*func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
            ),
            (
                "struct",
                r"^\s*type\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+struct",
            ),
        ],
        "c" | "cpp" => vec![
            (
                "class",
                r"^\s*(?:class|struct)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "function",
                r"^\s*(?:[A-Za-z_][A-Za-z0-9_:<>\*\&\s]+)\s+(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\s*\([^;]*\)\s*\{?\s*$",
            ),
        ],
        _ => Vec::new(),
    };
    specs
        .into_iter()
        .filter_map(|(k, p)| Regex::new(p).ok().map(|r| (k, r)))
        .collect()
}

fn find_symbol_end(lines: &[&str], start_idx: usize, language: &str) -> i32 {
    if language == "python" {
        return find_python_symbol_end(lines, start_idx);
    }

    let mut depth = 0i32;
    let mut saw_open = false;
    for (i, line) in lines.iter().enumerate().skip(start_idx) {
        for ch in line.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    saw_open = true;
                }
                '}' => {
                    depth -= 1;
                    if saw_open && depth <= 0 {
                        return i as i32 + 1;
                    }
                }
                _ => {}
            }
        }
        if !saw_open && i > start_idx && line.trim_end().ends_with(';') {
            return i as i32 + 1;
        }
    }
    lines.len() as i32
}

fn find_python_symbol_end(lines: &[&str], start_idx: usize) -> i32 {
    let start_indent = indent_width(lines[start_idx]);
    let mut signature_depth = 0i32;
    let mut body_started = false;
    let mut last_non_empty_line = start_idx as i32 + 1;

    for (i, line) in lines.iter().enumerate().skip(start_idx) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        for ch in line.chars() {
            match ch {
                '(' | '[' | '{' => signature_depth += 1,
                ')' | ']' | '}' => signature_depth -= 1,
                _ => {}
            }
        }

        if !body_started {
            if i > start_idx && signature_depth <= 0 && trimmed.ends_with(':') {
                body_started = true;
            } else if i == start_idx && signature_depth <= 0 && trimmed.ends_with(':') {
                body_started = true;
            }
            continue;
        }

        if indent_width(line) <= start_indent {
            return last_non_empty_line.max(start_idx as i32 + 1);
        }
        last_non_empty_line = i as i32 + 1;
    }

    lines.len() as i32
}

fn extract_call_names(body: &str, language: &str) -> BTreeSet<String> {
    let pattern = match language {
        "rust" | "go" | "c" | "cpp" | "typescript" | "javascript" | "python" => {
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
        }
        _ => return BTreeSet::new(),
    };
    let Ok(re) = Regex::new(pattern) else {
        return BTreeSet::new();
    };
    let keywords = [
        "if", "for", "while", "match", "switch", "return", "sizeof", "catch", "function",
    ];
    re.captures_iter(body)
        .filter_map(|c| c.name("name").map(|m| m.as_str().to_string()))
        .filter(|name| !keywords.contains(&name.as_str()))
        .collect()
}

struct ArtifactAnchors<'a> {
    by_file: BTreeMap<&'a str, Vec<&'a CodeArtifact>>,
}

impl<'a> ArtifactAnchors<'a> {
    fn new(artifacts: &'a [CodeArtifact]) -> Self {
        let mut by_file: BTreeMap<&'a str, Vec<&'a CodeArtifact>> = BTreeMap::new();
        for a in artifacts {
            by_file.entry(&a.file_path).or_default().push(a);
        }
        Self { by_file }
    }

    fn file_has_anchor(&self, file_path: &str) -> bool {
        self.by_file.contains_key(file_path)
    }

    fn match_symbol(&self, symbol: &CodebaseSymbol) -> Option<&'a CodeArtifact> {
        self.by_file
            .get(symbol.file_path.as_str())
            .and_then(|items| {
                items.iter().copied().find(|a| {
                    ranges_overlap(symbol.start_line, symbol.end_line, a.start_line, a.end_line)
                })
            })
    }
}

fn ranges_overlap(a_start: i32, a_end: i32, b_start: i32, b_end: i32) -> bool {
    a_start <= b_end && b_start <= a_end
}

/// Hard-skip: directories that never contain indexable source. Walked past entirely.
fn should_skip_path(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    parts.iter().any(|p| {
        matches!(
            *p,
            ".git"
                | "target"
                | "node_modules"
                | "dist"
                | "build"
                | ".next"
                | ".venv"
                | "venv"
                | "__pycache__"
        )
    })
}

/// Vendored: scanned but excluded from the main extraction pass. A later pass
/// reads them and keeps only symbols whose names are targeted by edges from
/// non-vendored code — so a `calls seal::encrypt` from main brings in the
/// matching SEAL symbol without dragging the whole vendored tree into the graph.
fn is_vendored_path(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    parts.iter().any(|p| {
        matches!(
            *p,
            "vendor"
                | "thirdparty"
                | "third_party"
                | "3rdparty"
                | "external"
                | "externals"
                | "deps"
                | "submodules"
        )
    })
}

struct IgnoreRules {
    suffixes: Vec<String>,
}

impl IgnoreRules {
    fn load(repo_root: &Path) -> Self {
        let path = repo_root.join(".gitignore");
        let text = fs::read_to_string(path).unwrap_or_default();
        let suffixes = text
            .lines()
            .map(str::trim)
            .filter(|l| {
                !l.is_empty() && !l.starts_with('#') && !l.contains('*') && !l.starts_with('!')
            })
            .map(|l| l.trim_start_matches('/').trim_end_matches('/').to_string())
            .collect();
        Self { suffixes }
    }

    fn matches(&self, rel: &str) -> bool {
        self.suffixes.iter().any(|s| {
            rel == s || rel.starts_with(&format!("{s}/")) || rel.ends_with(&format!("/{s}"))
        })
    }
}

fn read_repo_file(repo_root: &Path, rel: &str) -> Result<String, IndexerError> {
    let path = repo_root.join(rel);
    let root = repo_root
        .canonicalize()
        .map_err(|e| IndexerError::Io(repo_root.display().to_string(), e))?;
    let canonical = path
        .canonicalize()
        .map_err(|e| IndexerError::Io(path.display().to_string(), e))?;
    if !canonical.starts_with(root) {
        return Err(IndexerError::PathEscape(rel.to_string()));
    }
    fs::read_to_string(&canonical).map_err(|e| IndexerError::Io(rel.to_string(), e))
}

fn relative_path(root: &Path, path: &Path) -> Result<String, IndexerError> {
    let rel = path
        .strip_prefix(root)
        .map_err(|_| IndexerError::PathEscape(path.display().to_string()))?;
    Ok(rel.to_string_lossy().replace('\\', "/"))
}

fn slice_lines(content: &str, start_line: i32, end_line: i32) -> String {
    content
        .lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let line_no = i as i32 + 1;
            (line_no >= start_line && line_no <= end_line).then_some(line)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn bounded_file_text(content: &str) -> String {
    content.lines().take(250).collect::<Vec<_>>().join("\n")
}

fn indent_width(line: &str) -> usize {
    line.chars().take_while(|c| c.is_whitespace()).count()
}

fn estimate_tokens(text: &str) -> i32 {
    (text.len() / 4).max(1) as i32
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Normalise a pending-edge `target_name` into the key we used for
/// `symbol_by_name` lookups. For qualified names (`foo::bar`,
/// `self.helper`) we match on the tail component.
fn target_key(target: &str) -> &str {
    if let Some(tail) = target.rsplit_once("::") {
        tail.1
    } else if let Some(tail) = target.rsplit_once('.') {
        tail.1
    } else {
        target
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_analysis::{ArtifactStatus, MatchConfidence};
    use chrono::Utc;
    use tempfile::tempdir;

    #[test]
    fn builds_algorithm_focused_chunks_from_anchor_file() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("lib.rs"),
            r#"
pub fn helper(x: i32) -> i32 {
    x + 1
}

pub fn rebalance(values: &[i32]) -> i32 {
    helper(values.len() as i32)
}
"#,
        )
        .unwrap();

        let artifact = CodeArtifact {
            id: Uuid::new_v4(),
            tenant_id: Uuid::new_v4(),
            workspace_id: Uuid::new_v4(),
            document_id: "doc".to_string(),
            algorithm_id: Some(Uuid::new_v4()),
            document_repo_id: Uuid::new_v4(),
            repo_commit: "abc".to_string(),
            repo_license: Some("MIT".to_string()),
            language: "rust".to_string(),
            file_path: "src/lib.rs".to_string(),
            symbol_name: Some("rebalance".to_string()),
            start_line: 6,
            end_line: 8,
            snippet: "pub fn rebalance(values: &[i32]) -> i32 { helper(values.len() as i32) }"
                .to_string(),
            match_rationale: None,
            match_confidence: MatchConfidence::High,
            status: ArtifactStatus::Approved,
            embedding_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let out = ReferenceCodebaseIndexer::new(IndexLimits {
            max_files: 100,
            max_file_bytes: 100_000,
            max_chunks: 100,
        })
        .build(dir.path(), CodebaseIndexMode::AlgorithmFocused, &[artifact])
        .unwrap();

        assert_eq!(
            out.files
                .iter()
                .filter(|f| f.skipped_reason.is_none())
                .count(),
            1
        );
        assert!(out.symbols.iter().any(|s| s.name == "rebalance"));
        assert!(out
            .edges
            .iter()
            .any(|e| e.edge_type == "calls" && e.target_name.as_deref() == Some("helper")));
        assert!(out
            .chunks
            .iter()
            .any(|c| c.chunk_kind == "algorithm_anchor"
                && c.symbol_name.as_deref() == Some("rebalance")));
    }

    #[test]
    fn python_multiline_signature_extends_through_body() {
        let content = r#"def rebalance_clusters(
    vecs,
    centers,
    labels,
    cluster_bound,
):
    while True:
        labels = helper(labels)
        break
    return labels

def unrelated():
    return 1
"#;
        let lines = content.lines().collect::<Vec<_>>();
        assert_eq!(find_python_symbol_end(&lines, 0), 10);
    }
}
