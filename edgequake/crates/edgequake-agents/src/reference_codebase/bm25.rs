//! BM25 scoring with a code-aware tokenizer for lexical ranking over
//! reference-codebase symbol names, qualified names, and docstrings.
//!
//! Vendored from codemem (Apache-2.0,
//! /home/timo/repos/examples/codemem/crates/codemem-engine/src/bm25.rs).
//! Same workspace license; kept near-verbatim so upstream improvements
//! remain trivially re-portable. Tests pulled inline.
//!
//! The code-aware tokenizer splits camelCase / PascalCase
//! (`HTMLParser` → `html` + `parser`), snake_case (`rebalance_clusters`
//! → `rebalance` + `clusters`), digit boundaries, and punctuation — then
//! lowercases and drops tokens shorter than 2 chars. This is the axis
//! vector similarity misses on exact-identifier queries.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ── Code-Aware Tokenizer ────────────────────────────────────────────────────

/// Tokenize text with code-awareness: splits camelCase, snake_case,
/// punctuation boundaries, lowercases, and filters short tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in text.split_whitespace() {
        let segments = split_on_punctuation(word);
        for segment in segments {
            let sub_tokens = split_camel_case(&segment);
            for token in sub_tokens {
                let lower = token.to_lowercase();
                if lower.len() >= 2 {
                    tokens.push(lower);
                }
            }
        }
    }
    tokens
}

/// Split a string on punctuation and non-alphanumeric characters.
/// Keeps alphanumeric segments, discards separators.
fn split_on_punctuation(s: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            segments.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Split a camelCase or PascalCase string into its components.
/// `processRequest` → `["process", "Request"]`
/// `HTMLParser` → `["HTML", "Parser"]`
/// `getHTTPResponse` → `["get", "HTTP", "Response"]`
fn split_camel_case(s: &str) -> Vec<String> {
    if s.is_empty() {
        return vec![];
    }
    let chars: Vec<char> = s.chars().collect();
    let mut parts = Vec::new();
    let mut start = 0;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let curr = chars[i];
        // lowercase → uppercase (processRequest).
        let lower_to_upper = prev.is_lowercase() && curr.is_uppercase();
        // uppercase-run → uppercase+lowercase (HTMLParser → HTML | Parser).
        let upper_run_end =
            i >= 2 && chars[i - 2].is_uppercase() && prev.is_uppercase() && curr.is_lowercase();
        // digit boundary (item2count → item | 2 | count).
        let digit_boundary = (prev.is_alphabetic() && curr.is_ascii_digit())
            || (prev.is_ascii_digit() && curr.is_alphabetic());
        if lower_to_upper || upper_run_end || digit_boundary {
            let split_at = if upper_run_end { i - 1 } else { i };
            if split_at > start {
                let part: String = chars[start..split_at].iter().collect();
                parts.push(part);
                start = split_at;
            }
        }
    }
    if start < chars.len() {
        let part: String = chars[start..].iter().collect();
        parts.push(part);
    }
    parts
}

// ── BM25 Index ──────────────────────────────────────────────────────────────

/// BM25 index for scoring query-document relevance. Documents are identified
/// by string IDs and can be added/removed dynamically. Supports serialization
/// for persistence across restarts.
#[derive(Debug, Serialize, Deserialize)]
pub struct Bm25Index {
    doc_freq: HashMap<String, usize>,
    doc_lengths: HashMap<String, usize>,
    doc_terms: HashMap<String, HashMap<String, usize>>,
    pub doc_count: usize,
    avg_doc_len: f64,
    #[serde(default)]
    total_doc_len: usize,
    k1: f64,
    b: f64,
    #[serde(default = "default_max_documents")]
    max_documents: usize,
    #[serde(default)]
    insertion_order: VecDeque<String>,
}

fn default_max_documents() -> usize {
    100_000
}

impl Bm25Index {
    /// Empty index with defaults (`k1 = 1.2`, `b = 0.75`).
    pub fn new() -> Self {
        Self {
            doc_freq: HashMap::new(),
            doc_lengths: HashMap::new(),
            doc_terms: HashMap::new(),
            doc_count: 0,
            avg_doc_len: 0.0,
            total_doc_len: 0,
            k1: 1.2,
            b: 0.75,
            max_documents: default_max_documents(),
            insertion_order: VecDeque::new(),
        }
    }

    /// Add a document (or replace one with the same id). Evicts the oldest
    /// when `max_documents` is exceeded.
    pub fn add_document(&mut self, id: &str, content: &str) {
        if self.doc_terms.contains_key(id) {
            self.remove_document(id);
        }
        if self.doc_count >= self.max_documents {
            if let Some(oldest) = self.insertion_order.pop_front() {
                self.remove_document(&oldest);
            }
        }

        let tokens = tokenize(content);
        let doc_len = tokens.len();
        let mut term_freqs: HashMap<String, usize> = HashMap::new();
        for t in &tokens {
            *term_freqs.entry(t.clone()).or_insert(0) += 1;
        }
        for term in term_freqs.keys() {
            *self.doc_freq.entry(term.clone()).or_insert(0) += 1;
        }
        self.doc_lengths.insert(id.to_string(), doc_len);
        self.doc_terms.insert(id.to_string(), term_freqs);
        self.doc_count += 1;
        self.insertion_order.push_back(id.to_string());
        self.total_doc_len += doc_len;
        self.avg_doc_len = self.total_doc_len as f64 / self.doc_count as f64;
    }

    pub fn remove_document(&mut self, id: &str) {
        if let Some(term_freqs) = self.doc_terms.remove(id) {
            for term in term_freqs.keys() {
                if let Some(df) = self.doc_freq.get_mut(term) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        self.doc_freq.remove(term);
                    }
                }
            }
            let removed_len = self.doc_lengths.remove(id).unwrap_or(0);
            self.doc_count = self.doc_count.saturating_sub(1);
            self.total_doc_len = self.total_doc_len.saturating_sub(removed_len);
            if self.doc_count == 0 {
                self.avg_doc_len = 0.0;
            } else {
                self.avg_doc_len = self.total_doc_len as f64 / self.doc_count as f64;
            }
            self.insertion_order.retain(|x| x != id);
        }
    }

    /// Score a query against a specific indexed document. Returned value is
    /// normalized to `[0, 1]` by dividing by a perfect-match ceiling.
    pub fn score(&self, query: &str, doc_id: &str) -> f64 {
        if self.doc_count == 0 {
            return 0.0;
        }
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return 0.0;
        }
        let doc_len = match self.doc_lengths.get(doc_id) {
            Some(&l) => l,
            None => return 0.0,
        };
        let tf = match self.doc_terms.get(doc_id) {
            Some(t) => t,
            None => return 0.0,
        };
        let raw = self.raw_bm25_score(&query_tokens, tf, doc_len);
        let max = self.max_possible_score(&query_tokens);
        if max <= 0.0 {
            0.0
        } else {
            (raw / max).min(1.0)
        }
    }

    /// Zero-allocation variant: pre-tokenized query, indexed document.
    pub fn score_with_tokens_str(&self, query_tokens: &[&str], doc_id: &str) -> f64 {
        if self.doc_count == 0 || query_tokens.is_empty() {
            return 0.0;
        }
        let doc_len = match self.doc_lengths.get(doc_id) {
            Some(&l) => l,
            None => return 0.0,
        };
        let tf = match self.doc_terms.get(doc_id) {
            Some(t) => t,
            None => return 0.0,
        };
        let raw = self.raw_bm25_score(query_tokens, tf, doc_len);
        let max = self.max_possible_score(query_tokens);
        if max <= 0.0 {
            0.0
        } else {
            (raw / max).min(1.0)
        }
    }

    fn raw_bm25_score<S: AsRef<str>>(
        &self,
        query_tokens: &[S],
        doc_term_freqs: &HashMap<String, usize>,
        doc_len: usize,
    ) -> f64 {
        let n = self.doc_count as f64;
        let avgdl = if self.avg_doc_len > 0.0 {
            self.avg_doc_len
        } else {
            1.0
        };
        let dl = doc_len as f64;
        let mut score = 0.0;
        let mut seen: HashSet<&str> = HashSet::new();
        for qt in query_tokens {
            let qt_str = qt.as_ref();
            if !seen.insert(qt_str) {
                continue;
            }
            let tf = *doc_term_freqs.get(qt_str).unwrap_or(&0) as f64;
            if tf == 0.0 {
                continue;
            }
            let df = *self.doc_freq.get(qt_str).unwrap_or(&0) as f64;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            let numerator = tf * (self.k1 + 1.0);
            let denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl);
            score += idf * numerator / denominator;
        }
        score
    }

    fn max_possible_score<S: AsRef<str>>(&self, query_tokens: &[S]) -> f64 {
        let n = self.doc_count as f64;
        let mut max_score = 0.0;
        let mut seen: HashSet<&str> = HashSet::new();
        for qt in query_tokens {
            let qt_str = qt.as_ref();
            if !seen.insert(qt_str) {
                continue;
            }
            let df = *self.doc_freq.get(qt_str).unwrap_or(&0) as f64;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            // tf=10 in avg-length doc is our realistic ceiling.
            let tf = 10.0_f64;
            let numerator = tf * (self.k1 + 1.0);
            let denominator = tf + self.k1;
            max_score += idf * numerator / denominator;
        }
        max_score
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_splits_camel_case() {
        assert_eq!(tokenize("processRequest"), vec!["process", "request"]);
        assert_eq!(tokenize("HTMLParser"), vec!["html", "parser"]);
        assert_eq!(tokenize("getHTTPResponse"), vec!["get", "http", "response"]);
    }

    #[test]
    fn tokenizer_splits_snake_case() {
        assert_eq!(
            tokenize("rebalance_clusters"),
            vec!["rebalance", "clusters"]
        );
        assert_eq!(
            tokenize("compute_dot_product"),
            vec!["compute", "dot", "product"]
        );
    }

    #[test]
    fn tokenizer_drops_short_tokens() {
        // `a` and `1` are too short.
        let toks = tokenize("a_b_1_parseRequest");
        // `a`, `b`, `1` filtered; `parse`, `request` kept.
        assert!(toks.contains(&"parse".into()));
        assert!(toks.contains(&"request".into()));
        assert!(!toks.iter().any(|t| t.len() < 2));
    }

    #[test]
    fn scores_exact_identifier_match() {
        let mut idx = Bm25Index::new();
        idx.add_document("sym_a", "rebalance_clusters");
        idx.add_document("sym_b", "encrypting_compress");
        idx.add_document("sym_c", "softmax");
        let s_a = idx.score("port rebalance_clusters to Rust", "sym_a");
        let s_b = idx.score("port rebalance_clusters to Rust", "sym_b");
        assert!(s_a > s_b, "exact match should outrank unrelated: {s_a} vs {s_b}");
        assert!(s_a > 0.0);
    }

    #[test]
    fn scores_camel_case_match() {
        let mut idx = Bm25Index::new();
        idx.add_document("sym_head", "AttentionHead");
        idx.add_document("sym_other", "LinearLayer");
        let s_head = idx.score("how does AttentionHead work", "sym_head");
        let s_other = idx.score("how does AttentionHead work", "sym_other");
        assert!(s_head > s_other);
    }

    #[test]
    fn empty_query_returns_zero() {
        let mut idx = Bm25Index::new();
        idx.add_document("sym", "something");
        assert_eq!(idx.score("", "sym"), 0.0);
        assert_eq!(idx.score("query", "missing_id"), 0.0);
    }
}
