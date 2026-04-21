//! Extract code-identifier candidates from a natural-language query so the
//! retrieval path can route directly to symbol matches before vector search.
//!
//! Agent queries like "port `rebalance_clusters` to Rust" or "what does
//! AttentionHead::forward do" carry the exact function name 60%+ of the
//! time; vector search can lose these when surrounding prose is thin.
//! Extracting the identifier and pulling its chunks directly guarantees
//! the hit lands first in results.

use regex::Regex;
use std::collections::BTreeSet;
use std::sync::OnceLock;

fn camel() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    // PascalCase / CamelCase with at least two capitalised runs —
    // `AttentionHead`, `BatchEncoder`. Single-cap words like `Paper`
    // or `Matrix` don't match (too likely to be English).
    R.get_or_init(|| Regex::new(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-zA-Z0-9]*)+\b").unwrap())
}

fn snake() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    // snake_case with ≥1 underscore. `rebalance_clusters`,
    // `compute_dot_product`. Drops single-word lowercase (which would
    // false-positive on every English word).
    R.get_or_init(|| Regex::new(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b").unwrap())
}

fn qualified() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    // `seal::Encryptor::encrypt`, `torch.nn.Linear`, `a.b.c`.
    R.get_or_init(|| {
        Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:(?:::|\.)[a-zA-Z_][a-zA-Z0-9_]*)+\b").unwrap()
    })
}

fn backtick() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"`([^`]+)`").unwrap())
}

/// Parse the query text for identifier candidates. Returns both the raw
/// extracted strings AND the tail of each qualified path — so
/// `seal::Encryptor::encrypt` contributes `"seal::Encryptor::encrypt"` and
/// `"encrypt"`, letting the SQL lookup match on the short name too.
///
/// Candidates aren't validated here — the caller filters against the
/// symbol table, so a generous extractor that over-collects is fine.
pub fn extract_code_entities(query: &str) -> Vec<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    for m in camel().find_iter(query) {
        out.insert(m.as_str().to_string());
    }
    for m in snake().find_iter(query) {
        out.insert(m.as_str().to_string());
    }
    for m in qualified().find_iter(query) {
        let s = m.as_str();
        out.insert(s.to_string());
        // Tail after the last `::` or `.` — matches short `symbol_name` rows.
        let tail = s
            .rsplit_once("::")
            .map(|(_, t)| t)
            .or_else(|| s.rsplit_once('.').map(|(_, t)| t));
        if let Some(t) = tail {
            if !t.is_empty() {
                out.insert(t.to_string());
            }
        }
    }
    for c in backtick().captures_iter(query) {
        if let Some(m) = c.get(1) {
            let s = m.as_str().trim();
            if !s.is_empty() {
                out.insert(s.to_string());
            }
        }
    }
    out.into_iter().filter(|s| s.len() >= 2).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_camel_case() {
        let r = extract_code_entities("explain AttentionHead forward pass");
        assert!(r.contains(&"AttentionHead".to_string()));
        // "Paper" (single-cap word) must NOT match.
        let r2 = extract_code_entities("read the Paper");
        assert!(!r2.iter().any(|s| s == "Paper"));
    }

    #[test]
    fn extracts_snake_case() {
        let r = extract_code_entities("port rebalance_clusters to Rust");
        assert!(r.contains(&"rebalance_clusters".to_string()));
        // Lowercase single word without underscore — must NOT match.
        let r2 = extract_code_entities("port this to rust");
        assert!(!r2.iter().any(|s| s == "port" || s == "rust"));
    }

    #[test]
    fn extracts_qualified_with_tail() {
        let r = extract_code_entities("show seal::Encryptor::encrypt internals");
        assert!(r.contains(&"seal::Encryptor::encrypt".to_string()));
        assert!(r.contains(&"encrypt".to_string()));

        let r = extract_code_entities("find torch.nn.Linear");
        assert!(r.contains(&"torch.nn.Linear".to_string()));
        assert!(r.contains(&"Linear".to_string()));
    }

    #[test]
    fn extracts_backtick() {
        let r = extract_code_entities("what does `compute_dot` do");
        assert!(r.contains(&"compute_dot".to_string()));
    }

    #[test]
    fn empty_query_returns_empty() {
        assert!(extract_code_entities("").is_empty());
        assert!(extract_code_entities("the quick brown fox").is_empty());
    }
}
