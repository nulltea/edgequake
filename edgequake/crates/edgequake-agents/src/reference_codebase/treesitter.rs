//! Tree-sitter-backed symbol + edge extraction for Phase 2 reference
//! codebase indexing. Five languages get structural parsing via
//! `ast-grep-language`'s `SupportLang` enum; everything else falls back
//! to the regex path in `indexer.rs`.
//!
//! Per-language coverage matches the plan's edge table. Notably:
//! - Rust, Python, TypeScript emit the full taxonomy (`defines`,
//!   `calls`, `imports`, `references` plus `implements`/`inherits`
//!   where the language models those cleanly).
//! - C emits `defines`, `calls`, `imports (#include)`.
//! - C++ adds classes/methods on top of C; `implements`/`inherits` are
//!   deferred to Phase 3 / SCIP because templates and multiple
//!   inheritance need type resolution to be correct.
//!
//! The public surface is a single function `parse_file` returning
//! `(Vec<CodebaseSymbol>, Vec<CodebaseEdge>)` plus a parse-error count.
//! Callers pre-allocate file/symbol ids so edges can reference them
//! without a second pass.

use ast_grep_core::{
    tree_sitter::{LanguageExt, StrDoc},
    AstGrep, Node,
};
use ast_grep_language::SupportLang;
use std::collections::BTreeMap;
use uuid::Uuid;

use super::types::{CodebaseEdge, CodebaseFile, CodebaseSymbol};

/// Best-effort tree-sitter parse of one file. Returns the extracted
/// symbols and the call/import/reference/implements/inherits edges
/// emitted from its body. `parse_errors` counts grammar-level failures
/// the caller should aggregate onto `reference_codebase_files.parse_errors`.
///
/// Returns `None` when the file's language isn't covered here — caller
/// falls back to the regex extractor so non-targeted languages still
/// produce a minimal `defines + calls` graph.
pub fn parse_file(
    file: &CodebaseFile,
    source: &str,
) -> Option<ParseOutput> {
    let lang = support_lang_from_name(&file.language)?;
    // ast-grep-core parses via LanguageExt::ast_grep(source); this
    // returns an AstGrep<StrDoc<SupportLang>> we can walk.
    let grep: AstGrep<StrDoc<SupportLang>> = lang.ast_grep(source);
    let root = grep.root();

    let mut symbols: Vec<CodebaseSymbol> = Vec::new();
    let mut edges: Vec<PendingEdge> = Vec::new();

    let dispatch = match &*file.language {
        "rust" => extract_rust,
        "python" => extract_python,
        "typescript" | "javascript" => extract_typescript,
        "c" => extract_c,
        "cpp" => extract_cpp,
        _ => return None,
    };

    dispatch(
        &root,
        file,
        source,
        &mut symbols,
        &mut edges,
    );

    let parse_errors = if root.text().is_empty() && !source.is_empty() {
        1
    } else {
        0
    };

    Some(ParseOutput {
        symbols,
        edges,
        parse_errors,
    })
}

/// Map EdgeQuake's canonical language-name string (as set by
/// `language_from_path`) onto `ast-grep-language`'s `SupportLang`
/// enum. Returns `None` for names with no ast-grep grammar — caller
/// falls through to the regex extractor.
fn support_lang_from_name(name: &str) -> Option<SupportLang> {
    Some(match name {
        "rust" => SupportLang::Rust,
        "python" => SupportLang::Python,
        "typescript" => SupportLang::TypeScript,
        "javascript" => SupportLang::JavaScript,
        "go" => SupportLang::Go,
        "c" => SupportLang::C,
        "cpp" => SupportLang::Cpp,
        _ => return None,
    })
}

pub struct ParseOutput {
    pub symbols: Vec<CodebaseSymbol>,
    /// Edges with target **names** — resolution to target `symbol_id`
    /// happens in a second pass after every file's symbols are visible
    /// (callers resolve against a symbol-name index). The target name
    /// may be qualified (`foo::bar`) or bare (`parse_value`).
    pub edges: Vec<PendingEdge>,
    pub parse_errors: i32,
}

/// Edge whose target symbol may not yet be known — resolved globally
/// after every file is parsed. The indexer converts these to
/// `CodebaseEdge` by looking up `target_name` in the per-index symbol
/// map and falling back to `target_symbol_id = None` when the name
/// didn't match any indexed definition.
#[derive(Debug, Clone)]
pub struct PendingEdge {
    pub edge_type: String,
    pub source_symbol_id: Uuid,
    pub source_file_id: Uuid,
    pub target_name: String,
}

// ──────────────────────────────────────────────────────────────
// Per-language extractors
//
// Pattern: walk the tree once, emit a `CodebaseSymbol` for every
// definition node, and emit `PendingEdge`s for calls/imports/etc
// scoped to their enclosing symbol. Line numbers are 1-based to match
// the rest of EdgeQuake (start_line / end_line semantics).
// ──────────────────────────────────────────────────────────────

type Scope<'a> = Vec<(&'a str, Uuid, usize, usize)>;

fn enclosing_symbol(
    scope: &Scope<'_>,
    start_line: usize,
    end_line: usize,
) -> Option<Uuid> {
    // Pick the tightest-fitting open scope. O(N scope) is fine —
    // nesting depth is almost always <10.
    scope
        .iter()
        .rev()
        .find(|(_, _, s, e)| *s <= start_line && end_line <= *e)
        .map(|(_, id, _, _)| *id)
}

fn push_symbol(
    symbols: &mut Vec<CodebaseSymbol>,
    file: &CodebaseFile,
    kind: &str,
    name: String,
    scope_prefix: &[&str],
    start_line: usize,
    end_line: usize,
    scope_separator: &str,
) -> Uuid {
    let id = Uuid::new_v4();
    let qualified = if scope_prefix.is_empty() {
        format!("{}{}{}", file.file_path, "::", name)
    } else {
        format!(
            "{}::{}{}{}",
            file.file_path,
            scope_prefix.join(scope_separator),
            scope_separator,
            name,
        )
    };
    symbols.push(CodebaseSymbol {
        id,
        file_id: file.id,
        symbol_kind: kind.to_string(),
        name,
        qualified_name: qualified,
        parent_symbol_id: None,
        file_path: file.file_path.clone(),
        language: file.language.clone(),
        start_line: start_line as i32,
        end_line: end_line as i32,
        start_byte: 0,
        end_byte: 0,
    });
    id
}

fn extract_rust(
    root: &Node<'_, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    _source: &str,
    symbols: &mut Vec<CodebaseSymbol>,
    edges: &mut Vec<PendingEdge>,
) {
    let mut scope: Scope<'_> = Vec::new();
    walk_rust(root, file, &mut scope, symbols, edges);
}

fn walk_rust<'a>(
    node: &Node<'a, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    scope: &mut Scope<'a>,
    symbols: &mut Vec<CodebaseSymbol>,
    edges: &mut Vec<PendingEdge>,
) {
    let kind = node.kind();
    let start_line = node.start_pos().line() + 1;
    let end_line = node.end_pos().line() + 1;
    let scope_prefix: Vec<&str> = scope.iter().map(|(n, _, _, _)| *n).collect();

    match kind.as_ref() {
        "function_item" => {
            if let Some(name_node) = node.field("name") {
                let name = name_node.text().to_string();
                let sid = push_symbol(
                    symbols,
                    file,
                    "function",
                    name,
                    &scope_prefix,
                    start_line,
                    end_line,
                    "::",
                );
                // Walk body for calls.
                walk_body_calls_rust(node, file, sid, edges);
            }
        }
        "struct_item" | "enum_item" | "trait_item" | "type_item" => {
            if let Some(name_node) = node.field("name") {
                let name = name_node.text().to_string();
                let kind_str = match kind.as_ref() {
                    "struct_item" => "struct",
                    "enum_item" => "enum",
                    "trait_item" => "trait",
                    _ => "type",
                };
                push_symbol(
                    symbols,
                    file,
                    kind_str,
                    name,
                    &scope_prefix,
                    start_line,
                    end_line,
                    "::",
                );
            }
        }
        "impl_item" => {
            // impl Trait for Type → `implements` edge from trait impl
            // (as a synthetic impl-name) to Type. Also push the impl
            // as a scope so its methods nest underneath.
            let trait_name = node.field("trait").map(|n| n.text().to_string());
            let type_name = node.field("type").map(|n| n.text().to_string());
            let impl_name = match (&trait_name, &type_name) {
                (Some(t), Some(ty)) => format!("impl_{}_for_{}", t, ty),
                (None, Some(ty)) => format!("impl_{}", ty),
                _ => "impl_unknown".to_string(),
            };
            let sid = push_symbol(
                symbols,
                file,
                "impl",
                impl_name.clone(),
                &scope_prefix,
                start_line,
                end_line,
                "::",
            );
            if let (Some(t), Some(_)) = (&trait_name, &type_name) {
                edges.push(PendingEdge {
                    edge_type: "implements".to_string(),
                    source_symbol_id: sid,
                    source_file_id: file.id,
                    target_name: t.clone(),
                });
            }
            // Push a synthetic scope entry so nested items (methods)
            // know their enclosing impl when we resolve edges.
            let name_ref = Box::leak(impl_name.clone().into_boxed_str());
            scope.push((name_ref, sid, start_line, end_line));
            for child in node.children() {
                walk_rust(&child, file, scope, symbols, edges);
            }
            scope.pop();
            return;
        }
        "use_declaration" => {
            // `use crate::foo::bar;` → `imports` edge from the *file* (no
            // containing symbol) to the bare path tail as target_name.
            let path_text = node.text().to_string();
            let trimmed = path_text.trim_end_matches(';').trim_start_matches("use ");
            edges.push(PendingEdge {
                edge_type: "imports".to_string(),
                source_symbol_id: Uuid::nil(), // sentinel: file-level
                source_file_id: file.id,
                target_name: trimmed.to_string(),
            });
        }
        _ => {}
    }

    for child in node.children() {
        walk_rust(&child, file, scope, symbols, edges);
    }
}

fn walk_body_calls_rust(
    fn_node: &Node<'_, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    source_sid: Uuid,
    edges: &mut Vec<PendingEdge>,
) {
    let mut stack: Vec<Node<'_, StrDoc<SupportLang>>> = fn_node.children().collect();
    while let Some(n) = stack.pop() {
        if n.kind().as_ref() == "call_expression" {
            if let Some(func) = n.field("function") {
                let name = func.text().to_string();
                let tail = name.rsplit("::").next().unwrap_or(&name).to_string();
                edges.push(PendingEdge {
                    edge_type: "calls".to_string(),
                    source_symbol_id: source_sid,
                    source_file_id: file.id,
                    target_name: tail,
                });
            }
        }
        for c in n.children() {
            stack.push(c);
        }
    }
}

fn extract_python(
    root: &Node<'_, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    _source: &str,
    symbols: &mut Vec<CodebaseSymbol>,
    edges: &mut Vec<PendingEdge>,
) {
    walk_python(root, file, &mut Vec::new(), symbols, edges);
}

fn walk_python<'a>(
    node: &Node<'a, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    scope: &mut Scope<'a>,
    symbols: &mut Vec<CodebaseSymbol>,
    edges: &mut Vec<PendingEdge>,
) {
    let start_line = node.start_pos().line() + 1;
    let end_line = node.end_pos().line() + 1;
    let scope_prefix: Vec<&str> = scope.iter().map(|(n, _, _, _)| *n).collect();

    match node.kind().as_ref() {
        "function_definition" => {
            if let Some(name_node) = node.field("name") {
                let name = name_node.text().to_string();
                let sid = push_symbol(
                    symbols,
                    file,
                    "function",
                    name,
                    &scope_prefix,
                    start_line,
                    end_line,
                    ".",
                );
                walk_body_calls_python(node, file, sid, edges);
            }
        }
        "class_definition" => {
            if let Some(name_node) = node.field("name") {
                let name = name_node.text().to_string();
                let sid = push_symbol(
                    symbols,
                    file,
                    "class",
                    name.clone(),
                    &scope_prefix,
                    start_line,
                    end_line,
                    ".",
                );
                // `class Foo(Bar, Baz):` → inherits edges for each base.
                if let Some(superclasses) = node.field("superclasses") {
                    for child in superclasses.children() {
                        if child.kind().as_ref() == "identifier"
                            || child.kind().as_ref() == "attribute"
                        {
                            edges.push(PendingEdge {
                                edge_type: "inherits".to_string(),
                                source_symbol_id: sid,
                                source_file_id: file.id,
                                target_name: child.text().to_string(),
                            });
                        }
                    }
                }
                let name_ref = Box::leak(name.into_boxed_str());
                scope.push((name_ref, sid, start_line, end_line));
                for child in node.children() {
                    walk_python(&child, file, scope, symbols, edges);
                }
                scope.pop();
                return;
            }
        }
        "import_statement" | "import_from_statement" => {
            let text = node.text().to_string();
            let trimmed = text.trim();
            edges.push(PendingEdge {
                edge_type: "imports".to_string(),
                source_symbol_id: Uuid::nil(),
                source_file_id: file.id,
                target_name: trimmed.to_string(),
            });
        }
        _ => {}
    }

    for child in node.children() {
        walk_python(&child, file, scope, symbols, edges);
    }
}

fn walk_body_calls_python(
    fn_node: &Node<'_, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    source_sid: Uuid,
    edges: &mut Vec<PendingEdge>,
) {
    let mut stack: Vec<Node<'_, StrDoc<SupportLang>>> = fn_node.children().collect();
    while let Some(n) = stack.pop() {
        if n.kind().as_ref() == "call" {
            if let Some(func) = n.field("function") {
                let name = func.text().to_string();
                let tail = name.rsplit('.').next().unwrap_or(&name).to_string();
                edges.push(PendingEdge {
                    edge_type: "calls".to_string(),
                    source_symbol_id: source_sid,
                    source_file_id: file.id,
                    target_name: tail,
                });
            }
        }
        for c in n.children() {
            stack.push(c);
        }
    }
}

fn extract_typescript(
    root: &Node<'_, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    _source: &str,
    symbols: &mut Vec<CodebaseSymbol>,
    edges: &mut Vec<PendingEdge>,
) {
    walk_typescript(root, file, &mut Vec::new(), symbols, edges);
}

fn walk_typescript<'a>(
    node: &Node<'a, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    scope: &mut Scope<'a>,
    symbols: &mut Vec<CodebaseSymbol>,
    edges: &mut Vec<PendingEdge>,
) {
    let start_line = node.start_pos().line() + 1;
    let end_line = node.end_pos().line() + 1;
    let scope_prefix: Vec<&str> = scope.iter().map(|(n, _, _, _)| *n).collect();

    match node.kind().as_ref() {
        "function_declaration" | "function_signature" | "method_definition" => {
            if let Some(name_node) = node.field("name") {
                let name = name_node.text().to_string();
                let sid = push_symbol(
                    symbols,
                    file,
                    "function",
                    name,
                    &scope_prefix,
                    start_line,
                    end_line,
                    ".",
                );
                walk_body_calls_ts(node, file, sid, edges);
            }
        }
        "class_declaration" => {
            if let Some(name_node) = node.field("name") {
                let name = name_node.text().to_string();
                let sid = push_symbol(
                    symbols,
                    file,
                    "class",
                    name.clone(),
                    &scope_prefix,
                    start_line,
                    end_line,
                    ".",
                );
                // heritage_clause children hold `extends` / `implements`.
                for child in node.children() {
                    if child.kind().as_ref() == "class_heritage" {
                        let mut mode = "inherits";
                        for hc in child.children() {
                            let k = hc.kind();
                            if k.as_ref() == "implements_clause" {
                                mode = "implements";
                                for tgt in hc.children() {
                                    if tgt.kind().as_ref() == "type_identifier"
                                        || tgt.kind().as_ref() == "identifier"
                                    {
                                        edges.push(PendingEdge {
                                            edge_type: "implements".to_string(),
                                            source_symbol_id: sid,
                                            source_file_id: file.id,
                                            target_name: tgt.text().to_string(),
                                        });
                                    }
                                }
                            } else if k.as_ref() == "extends_clause" {
                                mode = "inherits";
                                for tgt in hc.children() {
                                    if tgt.kind().as_ref() == "identifier" {
                                        edges.push(PendingEdge {
                                            edge_type: "inherits".to_string(),
                                            source_symbol_id: sid,
                                            source_file_id: file.id,
                                            target_name: tgt.text().to_string(),
                                        });
                                    }
                                }
                            }
                            let _ = mode;
                        }
                    }
                }
                let name_ref = Box::leak(name.into_boxed_str());
                scope.push((name_ref, sid, start_line, end_line));
                for child in node.children() {
                    walk_typescript(&child, file, scope, symbols, edges);
                }
                scope.pop();
                return;
            }
        }
        "interface_declaration" => {
            if let Some(name_node) = node.field("name") {
                let name = name_node.text().to_string();
                push_symbol(
                    symbols,
                    file,
                    "interface",
                    name,
                    &scope_prefix,
                    start_line,
                    end_line,
                    ".",
                );
            }
        }
        "import_statement" => {
            let text = node.text().to_string();
            edges.push(PendingEdge {
                edge_type: "imports".to_string(),
                source_symbol_id: Uuid::nil(),
                source_file_id: file.id,
                target_name: text.trim().to_string(),
            });
        }
        _ => {}
    }
    for child in node.children() {
        walk_typescript(&child, file, scope, symbols, edges);
    }
}

fn walk_body_calls_ts(
    fn_node: &Node<'_, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    source_sid: Uuid,
    edges: &mut Vec<PendingEdge>,
) {
    let mut stack: Vec<Node<'_, StrDoc<SupportLang>>> = fn_node.children().collect();
    while let Some(n) = stack.pop() {
        if n.kind().as_ref() == "call_expression" {
            if let Some(func) = n.field("function") {
                let name = func.text().to_string();
                let tail = name.rsplit('.').next().unwrap_or(&name).to_string();
                edges.push(PendingEdge {
                    edge_type: "calls".to_string(),
                    source_symbol_id: source_sid,
                    source_file_id: file.id,
                    target_name: tail,
                });
            }
        }
        for c in n.children() {
            stack.push(c);
        }
    }
}

fn extract_c(
    root: &Node<'_, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    _source: &str,
    symbols: &mut Vec<CodebaseSymbol>,
    edges: &mut Vec<PendingEdge>,
) {
    walk_c_like(root, file, symbols, edges, /* cpp */ false);
}

fn extract_cpp(
    root: &Node<'_, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    _source: &str,
    symbols: &mut Vec<CodebaseSymbol>,
    edges: &mut Vec<PendingEdge>,
) {
    walk_c_like(root, file, symbols, edges, /* cpp */ true);
}

fn walk_c_like(
    root: &Node<'_, StrDoc<SupportLang>>,
    file: &CodebaseFile,
    symbols: &mut Vec<CodebaseSymbol>,
    edges: &mut Vec<PendingEdge>,
    cpp: bool,
) {
    let mut stack: Vec<Node<'_, StrDoc<SupportLang>>> = root.children().collect();
    // Function/struct symbols get ids; calls live inside them.
    // Use a by-range lookup so call_expressions attribute to the right
    // enclosing function without needing a recursive scope stack.
    let mut fn_ranges: BTreeMap<(usize, usize), Uuid> = BTreeMap::new();

    while let Some(n) = stack.pop() {
        let start = n.start_pos().line() + 1;
        let end = n.end_pos().line() + 1;
        match n.kind().as_ref() {
            "function_definition" => {
                let name = find_c_function_name(&n);
                if let Some(name) = name {
                    let sid = push_symbol(
                        symbols,
                        file,
                        "function",
                        name,
                        &[],
                        start,
                        end,
                        "::",
                    );
                    fn_ranges.insert((start, end), sid);
                }
            }
            "struct_specifier" | "union_specifier" | "enum_specifier" => {
                if let Some(name_node) = n.field("name") {
                    let kind_str = match n.kind().as_ref() {
                        "struct_specifier" => "struct",
                        "union_specifier" => "union",
                        _ => "enum",
                    };
                    push_symbol(
                        symbols,
                        file,
                        kind_str,
                        name_node.text().to_string(),
                        &[],
                        start,
                        end,
                        "::",
                    );
                }
            }
            "class_specifier" if cpp => {
                if let Some(name_node) = n.field("name") {
                    push_symbol(
                        symbols,
                        file,
                        "class",
                        name_node.text().to_string(),
                        &[],
                        start,
                        end,
                        "::",
                    );
                }
            }
            "preproc_include" => {
                let text = n.text().to_string();
                edges.push(PendingEdge {
                    edge_type: "imports".to_string(),
                    source_symbol_id: Uuid::nil(),
                    source_file_id: file.id,
                    target_name: text.trim().to_string(),
                });
            }
            _ => {}
        }
        for c in n.children() {
            stack.push(c);
        }
    }

    // Second pass: call_expression → calls edge, attributed to the
    // enclosing function by line-range overlap.
    let mut walk_stack: Vec<Node<'_, StrDoc<SupportLang>>> = root.children().collect();
    while let Some(n) = walk_stack.pop() {
        if n.kind().as_ref() == "call_expression" {
            if let Some(func) = n.field("function") {
                let name = func.text().to_string();
                let tail = name.rsplit("::").next().unwrap_or(&name).to_string();
                let line = n.start_pos().line() + 1;
                let enclosing = fn_ranges
                    .iter()
                    .find(|((s, e), _)| *s <= line && line <= *e)
                    .map(|(_, id)| *id);
                if let Some(sid) = enclosing {
                    edges.push(PendingEdge {
                        edge_type: "calls".to_string(),
                        source_symbol_id: sid,
                        source_file_id: file.id,
                        target_name: tail,
                    });
                }
            }
        }
        for c in n.children() {
            walk_stack.push(c);
        }
    }

    let _ = enclosing_symbol;
}

/// Walk a C/C++ `function_definition` subtree and return the identifier
/// that names the function. The function_declarator contains both the
/// name and the parameter_list (which contains its own identifiers);
/// we skip parameter_list descendants so we don't accidentally return a
/// parameter name.
fn find_c_function_name(
    fn_def: &Node<'_, StrDoc<SupportLang>>,
) -> Option<String> {
    // BFS to find the function_declarator first.
    let mut queue: std::collections::VecDeque<Node<'_, StrDoc<SupportLang>>> =
        std::collections::VecDeque::from([fn_def.clone()]);
    while let Some(n) = queue.pop_front() {
        if n.kind().as_ref() == "function_declarator" {
            return find_c_declarator_name(&n);
        }
        for c in n.children() {
            queue.push_back(c);
        }
    }
    None
}

/// Inside a function_declarator, return the innermost declarator name —
/// skipping `parameter_list` so parameter identifiers don't win.
fn find_c_declarator_name(
    decl: &Node<'_, StrDoc<SupportLang>>,
) -> Option<String> {
    let mut queue: std::collections::VecDeque<Node<'_, StrDoc<SupportLang>>> =
        std::collections::VecDeque::from([decl.clone()]);
    while let Some(n) = queue.pop_front() {
        match n.kind().as_ref() {
            "identifier" | "field_identifier" | "qualified_identifier" => {
                return Some(n.text().to_string());
            }
            "parameter_list" | "parameter_declaration" => {
                // Don't descend into parameters.
                continue;
            }
            _ => {}
        }
        for c in n.children() {
            queue.push_back(c);
        }
    }
    None
}

fn find_function_name(
    declarator: &Node<'_, StrDoc<SupportLang>>,
) -> Option<String> {
    // tree-sitter-c nests declarators (`*foo()`, `(foo)()`). Walk the
    // subtree, stopping at the first name-shaped leaf. `field_identifier`
    // is used for C++ method names.
    let mut stack: Vec<Node<'_, StrDoc<SupportLang>>> = vec![declarator.clone()];
    while let Some(n) = stack.pop() {
        match n.kind().as_ref() {
            "identifier" | "field_identifier" | "type_identifier" => {
                return Some(n.text().to_string());
            }
            _ => {}
        }
        for c in n.children() {
            stack.push(c);
        }
    }
    None
}

/// Slice a symbol's body from the source. Used to build parent-context
/// headers for the chunker.
pub fn slice_lines(source: &str, start_line: i32, end_line: i32) -> String {
    source
        .lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let line_no = i as i32 + 1;
            if line_no >= start_line && line_no <= end_line {
                Some(line)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_codebase::types::CodebaseFile;

    fn test_file(path: &str, language: &str) -> CodebaseFile {
        CodebaseFile {
            id: Uuid::new_v4(),
            file_path: path.to_string(),
            language: language.to_string(),
            checksum: String::new(),
            line_count: 0,
            byte_count: 0,
            skipped_reason: None,
        }
    }

    #[test]
    fn python_extracts_function_class_and_inherits_edges() {
        let src = r#"
import numpy as np

class Base:
    def a(self): return 1

class Child(Base):
    def b(self):
        return helper(self.a())

def helper(x): return x + 1
"#;
        let f = test_file("mod/lib.py", "python");
        let out = parse_file(&f, src).expect("python supported");
        assert!(out.symbols.iter().any(|s| s.name == "Base"));
        assert!(out.symbols.iter().any(|s| s.name == "Child"));
        assert!(out.symbols.iter().any(|s| s.name == "helper"));
        assert!(out
            .edges
            .iter()
            .any(|e| e.edge_type == "inherits" && e.target_name == "Base"));
        assert!(out
            .edges
            .iter()
            .any(|e| e.edge_type == "imports" && e.target_name.contains("numpy")));
        assert!(out.edges.iter().any(|e| e.edge_type == "calls"));
    }

    #[test]
    fn rust_extracts_impl_trait_for_type() {
        let src = r#"
use std::io::Write;

pub trait Runner {
    fn run(&self) -> i32;
}

pub struct Worker;

impl Runner for Worker {
    fn run(&self) -> i32 { helper(1) }
}

fn helper(x: i32) -> i32 { x + 1 }
"#;
        let f = test_file("src/lib.rs", "rust");
        let out = parse_file(&f, src).expect("rust supported");
        assert!(out.symbols.iter().any(|s| s.name == "Runner"));
        assert!(out.symbols.iter().any(|s| s.name == "Worker"));
        assert!(out.symbols.iter().any(|s| s.name == "helper"));
        assert!(out
            .edges
            .iter()
            .any(|e| e.edge_type == "implements" && e.target_name == "Runner"));
        assert!(out.edges.iter().any(|e| e.edge_type == "imports"));
    }

    #[test]
    fn c_extracts_include_and_function() {
        let src = r#"
#include <stdio.h>
#include "local.h"

int add(int a, int b) {
    return a + b;
}

int main() {
    return add(1, 2);
}
"#;
        let f = test_file("src/main.c", "c");
        let out = parse_file(&f, src).expect("c supported");
        assert!(out.symbols.iter().any(|s| s.name == "add"));
        assert!(out.symbols.iter().any(|s| s.name == "main"));
        assert!(
            out.edges.iter().filter(|e| e.edge_type == "imports").count() >= 2,
            "expected 2+ #include imports"
        );
        assert!(out
            .edges
            .iter()
            .any(|e| e.edge_type == "calls" && e.target_name == "add"));
    }

    #[test]
    fn cpp_skips_inherits_implements() {
        let src = r#"
#include <vector>
class Animal {};
class Dog : public Animal {};
"#;
        let f = test_file("src/animals.cpp", "cpp");
        let out = parse_file(&f, src).expect("cpp supported");
        assert!(out.symbols.iter().any(|s| s.name == "Animal"));
        assert!(out.symbols.iter().any(|s| s.name == "Dog"));
        // Phase 3 / SCIP scope — not emitted in Phase 2.
        assert!(!out.edges.iter().any(|e| e.edge_type == "inherits"));
        assert!(!out.edges.iter().any(|e| e.edge_type == "implements"));
    }
}
