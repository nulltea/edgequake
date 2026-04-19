//! Deterministic LaTeX repair for LLM- and OCR-mangled math.
//!
//! This crate is the keystone of the algorithm-extraction LaTeX-robustness
//! work. It contains a small collection of pure-string, idempotent, regex-
//! or scanner-bounded transforms that together repair the corruption
//! patterns observed in real extractions of cryptographic papers through
//! the edgequake pipeline:
//!
//! 1. **JSON-escape collisions** — an LLM emits e.g. `"\\text{...}"` but
//!    serialises only `"\text{...}"` into a JSON string; the JSON parser
//!    then interprets `\t` as a tab character and the resulting Rust
//!    `String` literally contains `<TAB>ext{...}`. Same pattern for
//!    `\b`→`\x08`, `\f`→`\x0C`, `\r`, `\n`. No general JSON-repair crate
//!    catches this because by the time we see the bytes, the `\` is gone.
//!    [`reconstruct_escape_collisions`] rebuilds them using a small table
//!    of common LaTeX commands.
//!
//! 2. **GLM-OCR artefact** — the paper's uniform-random-sample notation
//!    `x \leftarrow $ \mathbb{Z}_{2^\ell}` uses a bare `$` as a sampling
//!    operator. GLM-OCR misreads it as a bold script S and emits
//!    `\mathbb{S}`. [`repair_sample_dollar`] rewrites `\mathbb{S}` back to
//!    `\$` but only in the sampling context (followed by `\mathbb{Z}`,
//!    `\{`, or a similar set/ring token) so it doesn't damage genuine
//!    script-S usage.
//!
//! 3. **Unbalanced `[[...]]`** — the LLM occasionally copies a nested
//!    double-bracket notation `[[v^*]]^A` as `[v^*]]^A` (missing outer
//!    `[[`). [`balance_double_brackets`] repairs this line-by-line when
//!    there's more `]]` than `[[` and a bare `[` that can be doubled.
//!
//! 4. **Orphan `$`** — opening math delimiter without matching close.
//!    [`balance_math_delimiters`] appends a closing `$` when a line has
//!    an odd count of unescaped `$`.
//!
//! 5. **Unbraced single-token super/sub-scripts** — `^\theta` instead of
//!    `^{\theta}`. KaTeX renders these fine in many cases but some
//!    renderers refuse. [`brace_single_token_scripts`] wraps them.
//!
//! Every transform is idempotent (`f(f(x)) == f(x)`), pure, and does not
//! panic on arbitrary input. [`repair_latex`] is the top-level pipeline
//! that runs them in a fixed order.

use regex::Regex;
use std::sync::OnceLock;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Apply every repair transform in order. Idempotent.
///
/// Callers that only want a subset (e.g. post-OCR, before the text has
/// ever been JSON-serialised) should call the individual functions.
pub fn repair_latex(s: &str) -> String {
    let s = reconstruct_escape_collisions(s);
    let s = strip_fake_latex_codefence(&s);
    let s = repair_sample_dollar(&s);
    let s = repair_missing_inner_bracket(&s);
    let s = balance_double_brackets(&s);
    let s = balance_math_delimiters(&s);
    brace_single_token_scripts(&s)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. JSON-escape collision reconstruction
// ─────────────────────────────────────────────────────────────────────────────

/// Map from JSON-escape byte → (initial letter, full command names).
///
/// When the LLM emits `\text` inside a JSON string and forgets to double
/// the backslash, the JSON parser interprets `\t` as a tab escape; the
/// resulting Rust `String` holds TAB + `ext{...}`. To reconstruct we need
/// to know both the initial letter (`t` for TAB, reattached as `\t`) and
/// the remainder we expect to see after the control char.
///
/// Order within each list is "longest first" so `textbf` matches before
/// `text`.
fn escape_collision_table() -> &'static [(char, &'static [&'static str])] {
    &[
        // \t = tab (0x09) — commands starting with `t`. Stored as full command;
        // we match `command[1..]` against the post-TAB input.
        (
            '\t',
            &[
                "textbf", "textit", "textrm", "text", "theta", "times", "top", "tau", "to",
            ],
        ),
        // \b = backspace (0x08) — commands starting with `b`
        (
            '\u{0008}',
            &[
                "begin",
                "bigoplus",
                "bigotimes",
                "beta",
                "boxed",
                "big",
                "bar",
            ],
        ),
        // \f = form-feed (0x0C) — commands starting with `f`
        ('\u{000C}', &["forall", "frac", "frown", "flat", "floor"]),
        // \n = line-feed (0x0A) — commands starting with `n`.
        // We gate on math-like context to avoid rewriting legitimate
        // paragraph breaks in prose.
        ('\n', &["nabla", "notin", "newline", "neq", "ne", "not"]),
        // \r = carriage-return (0x0D) — commands starting with `r`.
        // Also gated on math-like context.
        ('\r', &["rightarrow", "rightharpoon", "rangle", "rho", "rm"]),
        // \v = vertical tab (0x0B) — rare
        (
            '\u{000B}',
            &["varepsilon", "varphi", "varpi", "vec", "vert"],
        ),
    ]
}

/// Rebuild `\command` where the leading `\` was eaten by JSON-escape
/// decoding, producing a control character followed by the command stem.
///
/// Only operates on math-like strings by default (see [`is_math_like`]) to
/// minimise risk of rewriting legitimate tab/newline characters in prose.
///
/// Idempotent: on repaired input the control chars are gone, so nothing
/// matches.
pub fn reconstruct_escape_collisions(s: &str) -> String {
    // Fast path: no control chars at all, nothing to do.
    if !s
        .bytes()
        .any(|b| matches!(b, 0x08 | 0x09 | 0x0A | 0x0B | 0x0C | 0x0D))
    {
        return s.to_string();
    }

    // We only aggressively reconstruct in math-like strings. In prose, we
    // still repair tabs/backspaces/form-feeds followed by a known stem (the
    // collision table) because real prose never has those sequences — but
    // we explicitly skip `\n` and `\r` reconstruction outside math context
    // to preserve paragraph breaks.
    let aggressive = is_math_like(s);

    let table = escape_collision_table();
    let mut out = String::with_capacity(s.len() + 16);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        // Is this a collision-candidate control char?
        if let Some((_, commands)) = table.iter().find(|(ctrl, _)| *ctrl == c) {
            // Skip LF/CR reconstruction in prose context (they're legitimate line breaks).
            let is_line_break = matches!(c, '\n' | '\r');
            if is_line_break && !aggressive {
                out.push(c);
                i += 1;
                continue;
            }
            // After the control char we expect `command[1..]` (command name
            // minus its first letter, which is the one the JSON escape ate).
            let rest = &s[i + 1..];
            let matched = commands.iter().find_map(|cmd| {
                let suffix = &cmd[1..];
                if rest.starts_with(suffix) {
                    // Ensure the character AFTER the stem is not another
                    // letter, so we don't treat `\ttexture` (→ TAB+`exture`)
                    // as `\text` + `ure`.
                    let after = rest.as_bytes().get(suffix.len()).copied();
                    let is_command_boundary = match after {
                        None => true,
                        Some(b) => !b.is_ascii_alphabetic(),
                    };
                    if is_command_boundary {
                        return Some((cmd, suffix.len()));
                    }
                }
                None
            });
            if let Some((cmd, consumed)) = matched {
                out.push('\\');
                out.push_str(cmd);
                i += 1 + consumed;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Heuristic: does this string look like it contains LaTeX math? Used to
/// decide whether `\n`/`\r` collisions should be reconstructed (aggressive)
/// or preserved as real line breaks (prose).
///
/// "Math-like" = contains at least one `$` or at least one unescaped
/// `\<letter>` command — either the prose references math, or the string
/// is entirely math.
fn is_math_like(s: &str) -> bool {
    if s.contains('$') {
        return true;
    }
    // Any `\letter...` sequence (real LaTeX command, since we only care
    // about un-corrupted commands which survive the JSON round-trip as
    // literal `\\letter`).
    s.contains('\\') && s.bytes().any(|b| b == b'\\')
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. `\mathbb{S}` OCR artefact
// ─────────────────────────────────────────────────────────────────────────────

fn mathbb_s_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `\mathbb{S}` followed by (optional whitespace) and one of:
    //   \mathbb{Z}, \mathbb{R}, \mathbb{F}, \mathbb{N}, \{, [, a digit
    // These all signal "sampling from a set", i.e. the original paper had
    // a bare `$` as sampling operator. Any standalone `\mathbb{S}` (e.g. a
    // paper genuinely introducing script-S as a variable) is left alone.
    RE.get_or_init(|| {
        Regex::new(r"\\mathbb\{S\}(\s*)(\\mathbb\{[ZRNFZQ]\}|\\\{|\[|\d)")
            .expect("mathbb_s_regex compiles")
    })
}

fn mathbb_s_fragmented_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // GLM-OCR sometimes fragments the math block around `\mathbb{S}`, yielding
    // `$...\leftarrow$ $\mathbb{S}$ $\mathbb{Z}...$` instead of the intended
    // `$...\leftarrow\$ \mathbb{Z}...$`. Collapse the `$ $\mathbb{S}$ $`
    // interstitial back to `\$ ` and keep the next set-token intact.
    RE.get_or_init(|| {
        Regex::new(r"\$\s*\$\s*\\mathbb\{S\}\s*\$\s*\$\s*(\\mathbb\{[ZRNFZQ]\}|\\\{|\[|\d)")
            .expect("mathbb_s_fragmented_regex compiles")
    })
}

/// Rewrite the GLM-OCR artefact `\mathbb{S}` back to `\$` in uniform-sampling
/// contexts. Handles both the contiguous form (`\mathbb{S}\mathbb{Z}`) and the
/// fragmented form where intervening `$...$` OCR boundaries sit around the
/// bogus `\mathbb{S}`. Idempotent.
pub fn repair_sample_dollar(s: &str) -> String {
    // Fragmented form first so it doesn't over-consume into the simpler case.
    let s = mathbb_s_fragmented_regex().replace_all(s, r"\$ $1").into_owned();
    mathbb_s_regex().replace_all(&s, r"\$$1$2").into_owned()
}

// ─────────────────────────────────────────────────────────────────────────────
// 2b. Missing inner `]` on `[[...]^` and `[[x[i]]^` shapes
// ─────────────────────────────────────────────────────────────────────────────

fn missing_bracket_simple_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `[[<content>]^` where content has no `[` and no `]` — should be
    // `[[<content>]]^`. Example: `[[v]^A` → `[[v]]^A`.
    RE.get_or_init(|| {
        Regex::new(r"(\[\[[^\[\]]*)\](\^)").expect("missing_bracket_simple compiles")
    })
}

fn missing_bracket_indexed_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `[[<prefix>[<index>]]^` — a double-bracket notation wrapping an indexed
    // reference. The inner `[...]` consumes one closing bracket that should
    // have been outer. Example: `[[x[i]]^A` → `[[x[i]]]^A`.
    RE.get_or_init(|| {
        Regex::new(r"(\[\[[^\[\]]*\[[^\[\]]*\]\])(\^)")
            .expect("missing_bracket_indexed compiles")
    })
}

/// Repair the two OCR-induced "missing inner `]`" shapes that
/// [`balance_double_brackets`] can't spot because its heuristic uses the
/// `]]` substring count rather than tracking single-bracket balance.
///
/// Idempotent: after rewrite the single `]^` / `]]^` with missing outer
/// bracket becomes balanced and the regex no longer fires.
pub fn repair_missing_inner_bracket(s: &str) -> String {
    let s = missing_bracket_indexed_regex()
        .replace_all(s, "$1]$2")
        .into_owned();
    missing_bracket_simple_regex()
        .replace_all(&s, "$1]]$2")
        .into_owned()
}

// ─────────────────────────────────────────────────────────────────────────────
// 2c. GLM-OCR fake LaTeX code-fence wrapper around math blocks
// ─────────────────────────────────────────────────────────────────────────────

fn fake_latex_fence_open_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `$$```latex\\$$` — GLM-OCR wraps a `$$...$$` math block in a fake
    // triple-backtick latex code fence whose opening line ends with a
    // literal `\\` then `$$`. Collapse the whole marker to a single `$$`.
    RE.get_or_init(|| {
        Regex::new(r"\$\$```latex\\+\$\$").expect("fake_latex_fence_open compiles")
    })
}

fn fake_latex_fence_close_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Matching closing artefact ` ``` $$$$` (three backticks then four
    // dollar signs). Collapse to `$$`.
    RE.get_or_init(|| {
        Regex::new(r"```\$\$\$\$").expect("fake_latex_fence_close compiles")
    })
}

/// Strip the fake ```` ```latex ```` fence GLM-OCR occasionally wraps around
/// a `$$...$$` display-math block. Idempotent: after rewrite the markers
/// are gone and the regex no longer fires.
pub fn strip_fake_latex_codefence(s: &str) -> String {
    let s = fake_latex_fence_open_regex().replace_all(s, "$$$$").into_owned();
    fake_latex_fence_close_regex()
        .replace_all(&s, "$$$$")
        .into_owned()
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Unbalanced `[[...]]`
// ─────────────────────────────────────────────────────────────────────────────

/// When a line has more `]]` than `[[`, look for an orphan `[` (single
/// bracket that's not already doubled) earlier in the line and double it.
///
/// Conservative: if there's no orphan `[`, leave the line alone. If the
/// line is already balanced, leave it alone. Idempotent.
pub fn balance_double_brackets(s: &str) -> String {
    s.lines()
        .map(balance_double_brackets_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn balance_double_brackets_line(line: &str) -> String {
    let open_dbl = count_literal(line, "[[");
    let close_dbl = count_literal(line, "]]");
    if close_dbl <= open_dbl {
        return line.to_string();
    }
    let deficit = close_dbl - open_dbl;

    // Walk the line; find the first N `[` that are not part of `[[`.
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len() + deficit);
    let mut fixed = 0;
    let mut i = 0;
    while i < bytes.len() {
        let is_bracket = bytes[i] == b'[';
        let is_part_of_doubled = is_bracket && bytes.get(i + 1).is_some_and(|b| *b == b'[')
            || is_bracket && i > 0 && bytes[i - 1] == b'[';
        if is_bracket && !is_part_of_doubled && fixed < deficit {
            out.push_str("[[");
            fixed += 1;
        } else {
            out.push(bytes[i] as char);
        }
        i += 1;
    }
    out
}

fn count_literal(hay: &str, needle: &str) -> usize {
    let mut count = 0;
    let mut rem = hay;
    while let Some(pos) = rem.find(needle) {
        count += 1;
        rem = &rem[pos + needle.len()..];
    }
    count
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Orphan `$` math delimiters
// ─────────────────────────────────────────────────────────────────────────────

/// For each line: if the count of unescaped `$` is odd, append `$` at the
/// end. Treats `$$` as one "token" (display math). Idempotent.
pub fn balance_math_delimiters(s: &str) -> String {
    s.lines()
        .map(balance_math_delimiters_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn balance_math_delimiters_line(line: &str) -> String {
    let mut inline_count = 0;
    let mut display_count = 0;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'$' {
            // escaped \$
            i += 2;
            continue;
        }
        if bytes[i] == b'$' {
            if bytes.get(i + 1).copied() == Some(b'$') {
                display_count += 1;
                i += 2;
                continue;
            }
            inline_count += 1;
        }
        i += 1;
    }
    let inline_odd = inline_count % 2 == 1;
    let display_odd = display_count % 2 == 1;
    let mut out = line.to_string();
    if display_odd {
        out.push_str("$$");
    } else if inline_odd {
        out.push('$');
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Unbraced single-token super/sub-scripts
// ─────────────────────────────────────────────────────────────────────────────

fn script_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Match `^\command` or `_\command`. Trailing context is checked in the
    // replacement closure because the `regex` crate doesn't support
    // look-ahead.
    RE.get_or_init(|| Regex::new(r"([\^_])(\\[a-zA-Z]+)").expect("script_regex compiles"))
}

/// Wrap single-token LaTeX commands in braces when used as super- or
/// sub-script operands. Idempotent — already-braced operands don't match
/// (the regex requires `^` or `_` immediately followed by `\`, but braced
/// forms have `{` between them).
pub fn brace_single_token_scripts(s: &str) -> String {
    let re = script_regex();
    // Use replace_all with a closure so we can look at what follows the
    // matched command and skip rewrites that would break legitimate input.
    let mut out = String::with_capacity(s.len());
    let mut last_end = 0;
    for caps in re.captures_iter(s) {
        let whole = caps.get(0).unwrap();
        let start = whole.start();
        let end = whole.end();
        // Look at the character immediately after the match. Skip rewrite
        // when the command is already followed by `{...}` (braced arg) or
        // by another alphabetic letter (we matched a prefix of a longer
        // token — shouldn't normally happen since we greedy-match `[a-zA-Z]+`,
        // but be defensive).
        let next_char = s[end..].chars().next();
        let skip = match next_char {
            Some('{') => true,
            Some(ch) if ch.is_ascii_alphabetic() => true,
            _ => false,
        };
        // Append the gap verbatim.
        out.push_str(&s[last_end..start]);
        if skip {
            out.push_str(&s[start..end]);
        } else {
            out.push_str(&caps[1]);
            out.push('{');
            out.push_str(&caps[2]);
            out.push('}');
        }
        last_end = end;
    }
    out.push_str(&s[last_end..]);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── reconstruct_escape_collisions ─────────────────────────────────────

    #[test]
    fn reconstructs_tab_text() {
        // Real corruption captured from B2A.pdf extraction.
        let corrupted = "$\\alpha \\leftarrow \text{Prg}(R,\\ell)$";
        let repaired = reconstruct_escape_collisions(corrupted);
        assert!(repaired.contains("\\text{Prg}"), "got: {repaired}");
        assert!(!repaired.contains('\t'));
    }

    #[test]
    fn reconstructs_backspace_beta() {
        let corrupted = "$\u{0008}eta_i = m_i$";
        let repaired = reconstruct_escape_collisions(corrupted);
        assert_eq!(repaired, "$\\beta_i = m_i$");
    }

    #[test]
    fn reconstructs_formfeed_frac() {
        let corrupted = "$\u{000C}rac{a}{b}$";
        let repaired = reconstruct_escape_collisions(corrupted);
        assert_eq!(repaired, "$\\frac{a}{b}$");
    }

    #[test]
    fn preserves_real_tab_in_prose() {
        // Prose with a real tab (not a collision), no math → leave alone.
        let input = "Step 1:\tdo this\nStep 2:\tdo that";
        let repaired = reconstruct_escape_collisions(input);
        assert_eq!(repaired, input);
    }

    #[test]
    fn preserves_newlines_in_prose() {
        // Newlines in prose without `$` should survive.
        let input = "line one\nline two\nline three";
        let repaired = reconstruct_escape_collisions(input);
        assert_eq!(repaired, input);
    }

    #[test]
    fn reconstructs_tab_textbf_greedy() {
        // Must match `textbf` (longer) before `text` (shorter).
        let corrupted = "$\textbf{bold}$";
        let repaired = reconstruct_escape_collisions(corrupted);
        assert!(repaired.contains("\\textbf{bold}"), "got: {repaired}");
    }

    #[test]
    fn does_not_over_match_texture() {
        // `\ttexture` should NOT be rewritten as `\text` + `ure`.
        let corrupted = "$\texture$";
        let repaired = reconstruct_escape_collisions(corrupted);
        // The `\t` is followed by `exture` which starts with "ext" but then
        // has "ure" — because `text` is in our table, we'd rewrite it as
        // `\text` + `ure`. That's actually wrong for "texture". The safeguard
        // is the command-boundary check: after `text` we expect non-alpha,
        // but `u` is alpha, so we DO skip the rewrite. Confirm:
        assert_eq!(
            repaired, corrupted,
            "must preserve 'texture', got: {repaired:?}"
        );
    }

    #[test]
    fn reconstruct_is_idempotent() {
        let input = "$\text{Prg}$ and $\u{0008}eta$";
        let once = reconstruct_escape_collisions(input);
        let twice = reconstruct_escape_collisions(&once);
        assert_eq!(once, twice);
    }

    // ─── repair_sample_dollar ─────────────────────────────────────────────

    #[test]
    fn repairs_mathbb_s_before_mathbb_z() {
        let corrupted = r"v_0, v_1 \leftarrow \mathbb{S} \mathbb{Z}_{2^{\ell}}^{2}";
        let repaired = repair_sample_dollar(corrupted);
        assert!(repaired.contains(r"\$"), "got: {repaired}");
        assert!(!repaired.contains(r"\mathbb{S}"));
    }

    #[test]
    fn preserves_mathbb_s_in_regular_context() {
        // Paper genuinely using script-S as a variable: `let S = \mathbb{S}`.
        // Our pattern only fires when S is followed by sampling context.
        let input = r"let $X = \mathbb{S}$ denote the set";
        let repaired = repair_sample_dollar(input);
        assert_eq!(repaired, input);
    }

    #[test]
    fn mathbb_s_is_idempotent() {
        let input = r"v \leftarrow \mathbb{S} \mathbb{Z}_{p}";
        let once = repair_sample_dollar(input);
        let twice = repair_sample_dollar(&once);
        assert_eq!(once, twice);
    }

    // ─── balance_double_brackets ──────────────────────────────────────────

    #[test]
    fn balances_missing_outer_open() {
        // Captured verbatim from Protocol 1 step 9 corruption.
        let input = r"$[[b]]^A \leftarrow \text{RSS.add}([v^*]]^A, [[\beta]]^A)$";
        let repaired = balance_double_brackets(input);
        assert!(repaired.contains("[[v^*]]"), "got: {repaired}");
    }

    #[test]
    fn leaves_balanced_alone() {
        let input = r"[[x]]^A = [[y]]^A + [[z]]^A";
        assert_eq!(balance_double_brackets(input), input);
    }

    #[test]
    fn balance_brackets_is_idempotent() {
        let input = r"$[v^*]]^A, [[\beta]]^A$";
        let once = balance_double_brackets(input);
        let twice = balance_double_brackets(&once);
        assert_eq!(once, twice);
    }

    // ─── balance_math_delimiters ──────────────────────────────────────────

    #[test]
    fn closes_orphan_inline_dollar() {
        let input = "text $a + b and more";
        let repaired = balance_math_delimiters(input);
        assert_eq!(repaired, "text $a + b and more$");
    }

    #[test]
    fn closes_orphan_display_dollars() {
        let input = "text $$a + b and more";
        let repaired = balance_math_delimiters(input);
        assert_eq!(repaired, "text $$a + b and more$$");
    }

    #[test]
    fn leaves_balanced_dollars_alone() {
        let input = "some $inline$ and $$display$$ math";
        assert_eq!(balance_math_delimiters(input), input);
    }

    #[test]
    fn ignores_escaped_dollar() {
        let input = r"price \$10 only";
        assert_eq!(balance_math_delimiters(input), input);
    }

    #[test]
    fn delimiter_balance_is_idempotent() {
        let input = "text $a + b";
        let once = balance_math_delimiters(input);
        let twice = balance_math_delimiters(&once);
        assert_eq!(once, twice);
    }

    // ─── brace_single_token_scripts ───────────────────────────────────────

    #[test]
    fn braces_superscript_theta() {
        let input = r"$(-1)^\theta \cdot \alpha$";
        let repaired = brace_single_token_scripts(input);
        assert_eq!(repaired, r"$(-1)^{\theta} \cdot \alpha$");
    }

    #[test]
    fn braces_subscript_alpha() {
        let input = r"$x_\alpha + y$";
        let repaired = brace_single_token_scripts(input);
        assert_eq!(repaired, r"$x_{\alpha} + y$");
    }

    #[test]
    fn leaves_braced_scripts_alone() {
        let input = r"$x^{\theta} \cdot y_{\alpha}$";
        assert_eq!(brace_single_token_scripts(input), input);
    }

    #[test]
    fn leaves_plain_char_scripts_alone() {
        // We only brace `^\command`, not `^x` (plain chars). KaTeX accepts
        // `x^2` without braces and we shouldn't change it.
        let input = "$x^2 + y_i$";
        assert_eq!(brace_single_token_scripts(input), input);
    }

    #[test]
    fn script_brace_is_idempotent() {
        let input = r"$x^\theta + y_\alpha$";
        let once = brace_single_token_scripts(input);
        let twice = brace_single_token_scripts(&once);
        assert_eq!(once, twice);
    }

    // ─── Top-level repair_latex (double-apply smoke test) ─────────────────

    #[test]
    fn repair_latex_is_idempotent() {
        // Pathological input with all 5 corruption categories.
        let input = concat!(
            "$\text{Prg}(\u{0008}eta) + x^\theta$ and ",
            r"$v \leftarrow \mathbb{S} \mathbb{Z}_p$ with ",
            r"$[v^*]]^A$ orphan $math "
        );
        let once = repair_latex(input);
        let twice = repair_latex(&once);
        assert_eq!(once, twice, "repair_latex must be idempotent");

        // Sanity checks on the repaired output.
        assert!(!once.contains('\t'), "no tab: {once:?}");
        assert!(!once.contains('\u{0008}'), "no backspace: {once:?}");
        assert!(once.contains(r"\text"), "reconstructed \\text: {once:?}");
        assert!(once.contains(r"\beta"), "reconstructed \\beta: {once:?}");
        assert!(once.contains("^{\\theta}"), "braced ^theta: {once:?}");
        assert!(
            !once.contains(r"\mathbb{S} \mathbb{Z}"),
            "mathbb-S removed: {once:?}"
        );
        assert!(once.contains("[[v^*]]"), "bracket balanced: {once:?}");
    }

    // ─── Fragmented `\mathbb{S}` (GLM-OCR broken math block) ────────────────

    #[test]
    fn fragmented_mathbb_s_is_repaired() {
        // Observed on B2A.pdf page 2 — OCR shattered the math block around
        // the sampling `\$`, leaving three fragmented `$...$` pairs.
        let input = r"$v_{0}, v_{1} \leftarrow$ $\mathbb{S}$ $\mathbb{Z}_{2^{\ell}}^{2}$";
        let out = repair_sample_dollar(input);
        assert_eq!(
            out,
            r"$v_{0}, v_{1} \leftarrow\$ \mathbb{Z}_{2^{\ell}}^{2}$"
        );
        // Idempotent.
        assert_eq!(repair_sample_dollar(&out), out);
    }

    #[test]
    fn contiguous_mathbb_s_still_works() {
        let input = r"$v \leftarrow \mathbb{S} \mathbb{Z}_p$";
        let out = repair_sample_dollar(input);
        assert!(out.contains(r"\$"), "got {out:?}");
        assert!(!out.contains(r"\mathbb{S}"), "got {out:?}");
    }

    #[test]
    fn genuine_script_s_is_preserved() {
        // Paper genuinely introduces `\mathbb{S}` as a set symbol — not
        // followed by another mathbb/brace/digit, so we leave it alone.
        let input = r"Let $\mathbb{S}$ be the set of all strategies.";
        assert_eq!(repair_sample_dollar(input), input);
    }

    // ─── Missing inner bracket repairs ──────────────────────────────────────

    #[test]
    fn simple_missing_inner_bracket_repaired() {
        // `[[v]^A` → `[[v]]^A`
        let out = repair_missing_inner_bracket(r"$$[[v]^A \leftarrow \text{X}$$");
        assert_eq!(out, r"$$[[v]]^A \leftarrow \text{X}$$");
        // Idempotent.
        assert_eq!(repair_missing_inner_bracket(&out), out);
    }

    #[test]
    fn indexed_missing_inner_bracket_repaired() {
        // `[[x[i]]^A` → `[[x[i]]]^A`
        let out = repair_missing_inner_bracket(r"$$[[x[i]]^A \leftarrow \text{Y}$$");
        assert_eq!(out, r"$$[[x[i]]]^A \leftarrow \text{Y}$$");
        assert_eq!(repair_missing_inner_bracket(&out), out);
    }

    #[test]
    fn balanced_double_bracket_untouched() {
        // Already correct — must NOT add extra bracket.
        let input = r"$$[[v]]^A \leftarrow [[b]]^A$$";
        assert_eq!(repair_missing_inner_bracket(input), input);
    }

    #[test]
    fn balanced_indexed_bracket_untouched() {
        let input = r"$$[[x[i]]]^A \leftarrow [[y[j]]]^B$$";
        assert_eq!(repair_missing_inner_bracket(input), input);
    }

    // ─── Fake latex code-fence wrapper ──────────────────────────────────────

    #[test]
    fn strip_fake_latex_fence_round_trip() {
        let input = "before\n$$```latex\\\\$$\n[[v^*]]^A \\leftarrow X\\\\\n```$$$$\nafter";
        let out = strip_fake_latex_codefence(input);
        assert!(!out.contains("```latex"), "got {out:?}");
        assert!(!out.contains("$$$$"), "got {out:?}");
        // Idempotent.
        assert_eq!(strip_fake_latex_codefence(&out), out);
    }

    #[test]
    fn strip_fake_latex_fence_is_idempotent_on_clean_math() {
        let input = "$$\n\\beta_i = m_i \\oplus x_{i,2}\n$$";
        assert_eq!(strip_fake_latex_codefence(input), input);
    }
}
