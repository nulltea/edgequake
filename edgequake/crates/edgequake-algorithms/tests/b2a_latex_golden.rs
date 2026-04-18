//! Golden regression test for algorithm-extraction LaTeX repair.
//!
//! Hermetic — no live VLM, no database, no Postgres. Reads a fixture with
//! the exact kind of malformed JSON the extraction LLM emits (single-
//! backslash LaTeX commands, unbalanced `[[…]]`, orphan `$`, `\mathbb{S}`
//! OCR artefacts, bare `^\theta` super/sub-scripts) and asserts that every
//! math-bearing field of the deserialised `ExtractedAlgorithm` is clean
//! after the full `parse_json_response` + `repair_extraction` pipeline.
//!
//! The fixture mirrors real B2A.pdf extraction output I reviewed against
//! the paper — every corruption pattern in this file was observed in
//! production at least once.

use edgequake_algorithms::extractor::{parse_json_response, repair_extraction};
use edgequake_algorithms::types::AlgorithmExtractionOutput;

const FIXTURE: &str = include_str!("fixtures/b2a_corrupted_llm_output.json");

fn parsed_and_repaired() -> AlgorithmExtractionOutput {
    let mut parsed: AlgorithmExtractionOutput =
        parse_json_response(FIXTURE).expect("fixture must parse");
    repair_extraction(&mut parsed);
    parsed
}

/// Walk every `String` / `Option<String>` field that might contain LaTeX
/// and feed it to a visitor.
fn for_each_math_field<F: FnMut(&str, &str)>(out: &AlgorithmExtractionOutput, mut visit: F) {
    for algo in &out.algorithms {
        visit("name", &algo.name);
        visit("description", &algo.description);
        if let Some(s) = algo.mathematical_notation.as_deref() {
            visit("mathematical_notation", s);
        }
        if let Some(s) = algo.pseudocode.as_deref() {
            visit("pseudocode", s);
        }
        if let Some(s) = algo.complexity.as_deref() {
            visit("complexity", s);
        }
        for step in &algo.steps {
            visit("step.action", &step.action);
            visit("step.details", &step.details);
            if let Some(s) = step.math.as_deref() {
                visit("step.math", s);
            }
        }
        for io in algo.inputs.iter().chain(algo.outputs.iter()) {
            visit("io.name", &io.name);
            visit("io.type", &io.io_type);
            visit("io.description", &io.description);
        }
        for pre in &algo.preconditions {
            visit("precondition", pre);
        }
    }
}

/// No raw control characters anywhere in the output — these would indicate
/// an unresolved JSON-escape collision (e.g. `\t` swallowed into a tab
/// byte inside `\text{…}`).
#[test]
fn no_leftover_control_chars() {
    let out = parsed_and_repaired();
    for_each_math_field(&out, |field, s| {
        for (idx, c) in s.chars().enumerate() {
            let is_forbidden = matches!(c, '\u{0008}' | '\u{0009}' | '\u{000B}' | '\u{000C}');
            assert!(
                !is_forbidden,
                "field `{field}` byte {idx} is control char U+{:04X}: {s:?}",
                c as u32,
            );
        }
    });
}

/// Double-bracket notation `[[…]]` must be balanced in every math field.
/// Captured corruption: Protocol 1 step 9 emitted `[v^*]]^A` missing its
/// outer open.
#[test]
fn double_brackets_balanced() {
    let out = parsed_and_repaired();
    for_each_math_field(&out, |field, s| {
        let open = count_occurrences(s, "[[");
        let close = count_occurrences(s, "]]");
        assert_eq!(
            open, close,
            "field `{field}` has {open} `[[` but {close} `]]`: {s:?}"
        );
    });
}

/// Inline `$…$` and display `$$…$$` math delimiters must balance
/// per-line. An orphan opening delimiter would render the rest of the
/// line as math in the UI.
#[test]
fn math_delimiters_balanced() {
    let out = parsed_and_repaired();
    for_each_math_field(&out, |field, s| {
        for (line_num, line) in s.lines().enumerate() {
            let mut inline = 0usize;
            let mut display = 0usize;
            let bytes = line.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'$' {
                    i += 2;
                    continue;
                }
                if bytes[i] == b'$' {
                    if bytes.get(i + 1).copied() == Some(b'$') {
                        display += 1;
                        i += 2;
                        continue;
                    }
                    inline += 1;
                }
                i += 1;
            }
            assert_eq!(
                inline % 2,
                0,
                "field `{field}` line {line_num} has {inline} unescaped `$` (odd): {line:?}"
            );
            assert_eq!(
                display % 2,
                0,
                "field `{field}` line {line_num} has {display} `$$` (odd): {line:?}"
            );
        }
    });
}

/// The `\mathbb{S}` GLM-OCR artefact for uniform-sampling `$` must not
/// survive (when followed by a sampling context like `\mathbb{Z}` or `\{`).
#[test]
fn no_mathbb_s_in_sampling_context() {
    let out = parsed_and_repaired();
    for_each_math_field(&out, |field, s| {
        // The repair narrowly rewrites `\mathbb{S} \mathbb{Z}` or `\mathbb{S}
        // \{` — assert neither surface form survives.
        assert!(
            !s.contains(r"\mathbb{S} \mathbb{Z}"),
            "field `{field}` still has sampling-context \\mathbb{{S}}: {s:?}"
        );
        assert!(
            !s.contains(r"\mathbb{S}\{"),
            "field `{field}` still has \\mathbb{{S}}\\{{ sampling literal: {s:?}"
        );
    });
}

/// Single-token LaTeX super/sub-script operands must be braced. `^\theta`
/// alone renders inconsistently across LaTeX engines.
#[test]
fn single_token_scripts_braced() {
    let out = parsed_and_repaired();
    // Scan every string for `^\` or `_\` followed by letters but no
    // immediately enclosing `{`.
    for_each_math_field(&out, |field, s| {
        let re = regex::Regex::new(r"[\^_]\\[a-zA-Z]+").expect("regex");
        for m in re.find_iter(s) {
            // The match ends at the last letter of the command. Accept if
            // the character immediately after is `{` (command takes an arg)
            // or end-of-string; reject otherwise.
            let next_char = s[m.end()..].chars().next();
            let ok = match next_char {
                Some('{') => true,
                _ => false,
            };
            assert!(
                ok,
                "field `{field}` has unbraced script operand at match `{}` (next char: {next_char:?}): {s:?}",
                m.as_str()
            );
        }
    });
}

/// Every LaTeX math expression in repaired output must parse via
/// `pulldown-latex`. This is the strongest guarantee: not just "no
/// corruption indicators" but "actually parseable math".
#[test]
fn pulldown_latex_parses_math_fields() {
    let out = parsed_and_repaired();
    for_each_math_field(&out, |field, s| {
        // Only check fields that actually contain math (either `$…$` or
        // `\command`). Plain prose is allowed to not parse.
        if !s.contains('$') && !s.contains('\\') {
            return;
        }
        // Extract each `$…$` or `$$…$$` math span and parse.
        for span in extract_math_spans(s) {
            let storage = pulldown_latex::Storage::new();
            let config = pulldown_latex::RenderConfig::default();
            let parser = pulldown_latex::Parser::new(span, &storage);
            let mut out_buf = String::new();
            let res = pulldown_latex::push_mathml(&mut out_buf, parser, config);
            assert!(
                res.is_ok(),
                "field `{field}` math span {span:?} did not parse: {:?}\nfull: {s:?}",
                res.err()
            );
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn count_occurrences(hay: &str, needle: &str) -> usize {
    let mut count = 0;
    let mut rem = hay;
    while let Some(pos) = rem.find(needle) {
        count += 1;
        rem = &rem[pos + needle.len()..];
    }
    count
}

/// Extract the contents of each `$…$` or `$$…$$` span (without delimiters).
fn extract_math_spans(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'$' {
            i += 2;
            continue;
        }
        if bytes[i] == b'$' {
            let is_display = bytes.get(i + 1).copied() == Some(b'$');
            let delim_len = if is_display { 2 } else { 1 };
            let start = i + delim_len;
            // Find matching close.
            let mut j = start;
            let close_found = loop {
                if j >= bytes.len() {
                    break false;
                }
                if bytes[j] == b'\\' && j + 1 < bytes.len() && bytes[j + 1] == b'$' {
                    j += 2;
                    continue;
                }
                if bytes[j] == b'$' {
                    let close_display = bytes.get(j + 1).copied() == Some(b'$');
                    if is_display == close_display {
                        break true;
                    }
                }
                j += 1;
            };
            if close_found {
                out.push(&s[start..j]);
                i = j + delim_len;
            } else {
                break;
            }
        } else {
            i += 1;
        }
    }
    out
}
