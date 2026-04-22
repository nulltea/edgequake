//! Tuple-delimited extraction result parser (SOTA format).
//!
//! Parses extraction output in the format used by LightRAG:
//! ```text
//! entity<|#|>Name<|#|>TYPE<|#|>Description
//! relation<|#|>Source<|#|>Target<|#|>keywords<|#|>Description
//! <|COMPLETE|>
//! ```
//!
//! # WHY Tuple Format Over JSON
//!
//! The tuple-delimited format is significantly more robust for LLM outputs:
//!
//! 1. **Partial output recovery**: Valid lines parse independently even if
//!    the response is truncated.
//! 2. **No escaping issues**: No quotes, backslashes, or unicode escaping.
//! 3. **Line-by-line processing**: Enables streaming extraction.
//! 4. **Battle-tested**: Proven in the LightRAG paper with millions of extractions.

use std::sync::LazyLock;

use regex::Regex;

use super::super::normalizer::normalize_entity_name;
use super::super::{DEFAULT_COMPLETION_DELIMITER, DEFAULT_TUPLE_DELIMITER};
use crate::error::Result;
use crate::extractor::{ExtractedEntity, ExtractedRelationship, ExtractionResult};

// ── Structural-noise filters ────────────────────────────────────────────
//
// The LLM prompt (see `prompts::entity_extraction`) tells the model to
// skip document structure, formal labels, single-letter notation, and
// role-only protocol actors. It usually complies; sometimes it doesn't.
// These regexes are the deterministic belt that catches the regressions.

/// Matches pure structural / formal-label references:
/// `Table 1`, `Figure 3a`, `Theorem 2.1`, `Lemma A.3`, `Protocol 4`,
/// `Algorithm 5`, `Section 4.2`, `Appendix A`, `Equation (2)`,
/// `Chapter 5`, `Corollary 1`, `Proposition 3`, `Definition 2`,
/// `Claim A.1`, `Remark 5`, `Example 3`, `Case 3`, `Step 2`,
/// `Phase 1`, `Round 3`. Case-insensitive; tolerates a trailing dot
/// and common abbreviation dots (`Fig.`, `Sec.`, `Thm.`).
///
/// The locator half is deliberately tight — it must be a
/// numeric-shaped or single-letter label. That rejects false positives
/// like `Figure skating` (where the "locator" is a full word) or
/// `Table of Contents`.
static STRUCTURAL_LABEL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?ix)
        ^
        (?:
            table | figure | fig\.? | section | sec\.? | appendix | app\.? |
            chapter | ch\.? | equation | eq\.? | theorem | thm\.? | lemma |
            proof | algorithm | alg\.? | protocol | corollary | proposition |
            prop\.? | definition | def\.? | claim | remark | example | ex\.? |
            case | step | phase | round
        )
        \s*
        \(?
        (?:
            \d+ (?: \. \d+ )* [a-z]?    # 1, 1a, 2.1, 2.1.3, 3a
            |
            [A-Z] (?: \. \d+ (?: \. \d+ )* )?   # A, A.1, A.2.3
        )
        \)?
        \.?
        $
        ",
    )
    .expect("STRUCTURAL_LABEL_RE compiles")
});

/// Matches `§3`, `§ 3.2`, `§A.1` — the section-mark character followed
/// by a numeric / alphanumeric locator.
static SECTION_MARK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^§\s*[0-9A-Za-z.]+$").expect("SECTION_MARK_RE compiles"));

/// Matches a generic protocol-role name standing alone:
/// `Adversary`, `Adversary (A)`, `Challenger`, `Verifier (V)`, …
/// A proper name attached to the role (e.g. `Adversary Eve`) bypasses
/// this because the regex anchors the role word at end-of-string (the
/// optional suffix is strictly one parenthesised single letter).
static ROLE_ACTOR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(?:adversary|challenger|verifier|prover|simulator|client|server)(?:\s*\([A-Za-z]\))?$",
    )
    .expect("ROLE_ACTOR_RE compiles")
});

/// Return true when the raw entity name (pre-normalisation) should be
/// dropped as structural / non-entity noise. Run against the raw name
/// so the regexes can rely on natural casing and punctuation.
///
/// Rules, in declaration order — any match is a drop:
///
/// 1. Single-character names (`S`, `Q`, `x`, `π`, …). Digit-only names
///    get the same treatment on the basis that a bare integer is never
///    a useful entity.
/// 2. Pure structural label (`Table 1`, `Theorem 2.1`, `Protocol 4`, …).
/// 3. Section-mark locator (`§3`).
/// 4. Role-only protocol actor (`Adversary`, `Adversary (A)`, …).
fn is_structural_noise(raw_name: &str) -> bool {
    let trimmed = raw_name.trim();
    if trimmed.is_empty() {
        return false; // handled by the empty-name branch in the caller
    }

    if trimmed.chars().count() == 1 {
        return true;
    }

    STRUCTURAL_LABEL_RE.is_match(trimmed)
        || SECTION_MARK_RE.is_match(trimmed)
        || ROLE_ACTOR_RE.is_match(trimmed)
}

/// Parser for tuple-delimited extraction results (SOTA format).
///
/// Parses extraction output in the format:
/// ```text
/// entity<|#|>Name<|#|>TYPE<|#|>Description
/// relation<|#|>Source<|#|>Target<|#|>keywords<|#|>Description
/// <|COMPLETE|>
/// ```
#[derive(Debug, Clone)]
pub struct TupleParser {
    tuple_delimiter: String,
    completion_delimiter: String,
}

impl Default for TupleParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TupleParser {
    /// Create a new tuple parser with default delimiters.
    pub fn new() -> Self {
        Self {
            tuple_delimiter: DEFAULT_TUPLE_DELIMITER.to_string(),
            completion_delimiter: DEFAULT_COMPLETION_DELIMITER.to_string(),
        }
    }

    /// Create a parser with custom delimiters.
    pub fn with_delimiters(tuple: &str, completion: &str) -> Self {
        Self {
            tuple_delimiter: tuple.to_string(),
            completion_delimiter: completion.to_string(),
        }
    }

    /// Parse a response into an extraction result.
    pub fn parse(&self, response: &str, chunk_id: &str) -> Result<ExtractionResult> {
        let mut result = ExtractionResult::new(chunk_id);
        let mut parse_errors = 0u64;
        let mut dropped_structural_entities = 0u64;
        let mut dropped_structural_relationships = 0u64;
        // Entities the structural-noise filter rejected. Their normalized
        // names are recorded so we can also drop any relationship whose
        // endpoint references one of them — otherwise the graph keeps the
        // edge dangling into the void.
        let mut dropped_entity_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Check if the response is complete
        let is_complete = response.contains(&self.completion_delimiter);
        result
            .metadata
            .insert("is_complete".to_string(), serde_json::json!(is_complete));

        for line in response.lines() {
            let line = line.trim();
            if line.is_empty() || line == self.completion_delimiter {
                continue;
            }

            let parts: Vec<&str> = line.split(&self.tuple_delimiter).collect();

            if parts.is_empty() {
                continue;
            }

            match parts[0].trim().to_lowercase().as_str() {
                "entity" if parts.len() >= 4 => {
                    let raw_name = parts[1].trim();
                    let entity_type = parts[2].trim().to_uppercase();
                    let description = parts[3].trim();

                    // Skip entities with empty or whitespace-only names
                    if raw_name.is_empty() {
                        parse_errors += 1;
                        continue;
                    }

                    // Structural-noise filter. Runs on the raw (pre-
                    // normalisation) name so the regexes can rely on
                    // natural casing and punctuation like `§3`.
                    if is_structural_noise(raw_name) {
                        let normalized_name = normalize_entity_name(raw_name);
                        if !normalized_name.is_empty() {
                            dropped_entity_names.insert(normalized_name.clone());
                        }
                        dropped_structural_entities += 1;
                        tracing::debug!(
                            raw_name = %raw_name,
                            entity_type = %entity_type,
                            "Skipping structural-noise entity"
                        );
                        continue;
                    }

                    let normalized_name = normalize_entity_name(raw_name);

                    // BR0006 defense: Skip entities that normalize to empty string
                    if normalized_name.is_empty() {
                        tracing::debug!(raw_name = %raw_name, "Skipping entity with empty normalized name");
                        continue;
                    }

                    result.add_entity(ExtractedEntity::new(
                        normalized_name,
                        entity_type,
                        description,
                    ));
                }
                "relation" | "relationship" if parts.len() >= 5 => {
                    let source = parts[1].trim();
                    let target = parts[2].trim();
                    let keywords_str = parts[3].trim();
                    let description = parts[4].trim();

                    // Drop relationships whose endpoint is a structural-
                    // noise entity. Checked on the raw name so the same
                    // regex catches `Table 1 → <real entity>` edges that
                    // leaked past the prompt.
                    if is_structural_noise(source) || is_structural_noise(target) {
                        dropped_structural_relationships += 1;
                        tracing::debug!(
                            raw_source = %source,
                            raw_target = %target,
                            "Skipping relationship with structural-noise endpoint"
                        );
                        continue;
                    }

                    let normalized_source = normalize_entity_name(source);
                    let normalized_target = normalize_entity_name(target);

                    // Also drop edges that reference an entity we already
                    // rejected earlier in the same chunk (belt-and-braces
                    // for cases where the prompt uses different casing).
                    if dropped_entity_names.contains(&normalized_source)
                        || dropped_entity_names.contains(&normalized_target)
                    {
                        dropped_structural_relationships += 1;
                        tracing::debug!(
                            source = %normalized_source,
                            target = %normalized_target,
                            "Skipping relationship referencing a previously-dropped entity"
                        );
                        continue;
                    }

                    // BR0006: Self-referencing relationships forbidden
                    if normalized_source == normalized_target {
                        tracing::debug!(
                            source = %normalized_source,
                            "Skipping self-referencing relationship (BR0006)"
                        );
                        continue;
                    }

                    // Skip relationships with empty normalized endpoints
                    if normalized_source.is_empty() || normalized_target.is_empty() {
                        tracing::debug!(
                            raw_source = %source,
                            raw_target = %target,
                            "Skipping relationship with empty normalized endpoint"
                        );
                        continue;
                    }

                    // BR0004: Parse and limit keywords (max 5 per edge)
                    let keywords: Vec<String> = keywords_str
                        .split(',')
                        .map(|k| k.trim().to_string())
                        .filter(|k| !k.is_empty())
                        .take(5)
                        .collect();

                    let mut rel = ExtractedRelationship::new(
                        normalized_source,
                        normalized_target,
                        keywords_str,
                    )
                    .with_description(description);

                    if !keywords.is_empty() {
                        rel = rel.with_keywords(keywords);
                    }

                    result.add_relationship(rel);
                }
                _ => {
                    // Unknown line type, count as parse error
                    if line.contains(&self.tuple_delimiter) {
                        parse_errors += 1;
                    }
                }
            }
        }

        result
            .metadata
            .insert("parser".to_string(), serde_json::json!("tuple"));
        result
            .metadata
            .insert("parse_errors".to_string(), serde_json::json!(parse_errors));
        result.metadata.insert(
            "dropped_structural_entities".to_string(),
            serde_json::json!(dropped_structural_entities),
        );
        result.metadata.insert(
            "dropped_structural_relationships".to_string(),
            serde_json::json!(dropped_structural_relationships),
        );

        Ok(result)
    }

    /// Check if the response appears complete.
    pub fn is_complete(&self, response: &str) -> bool {
        response.contains(&self.completion_delimiter)
    }
}

#[cfg(test)]
mod structural_filter_tests {
    use super::*;

    #[test]
    fn drops_numbered_structural_labels() {
        assert!(is_structural_noise("Table 1"));
        assert!(is_structural_noise("Figure 3"));
        assert!(is_structural_noise("Theorem 2.1"));
        assert!(is_structural_noise("Lemma A.3"));
        assert!(is_structural_noise("Protocol 4"));
        assert!(is_structural_noise("Section 4.2"));
        assert!(is_structural_noise("Algorithm 5"));
        assert!(is_structural_noise("Equation (2)"));
        assert!(is_structural_noise("Appendix A"));
        assert!(is_structural_noise("Corollary 1"));
        assert!(is_structural_noise("Definition 2"));
        // Abbreviated forms
        assert!(is_structural_noise("Fig. 3"));
        assert!(is_structural_noise("Sec. 4.1"));
        assert!(is_structural_noise("Thm. 2.1"));
    }

    #[test]
    fn drops_section_mark_refs() {
        assert!(is_structural_noise("§3"));
        assert!(is_structural_noise("§ 3.2"));
        assert!(is_structural_noise("§A.1"));
    }

    #[test]
    fn drops_single_letters_only() {
        assert!(is_structural_noise("S"));
        assert!(is_structural_noise("Q"));
        assert!(is_structural_noise("x"));
        assert!(is_structural_noise("π"));
        // 2+ chars are kept — real signal may exist in short notation,
        // and a short-all-caps allowlist/denylist is brittle.
        assert!(!is_structural_noise("P2"));
        assert!(!is_structural_noise("TLS"));
        assert!(!is_structural_noise("RFC"));
        assert!(!is_structural_noise("sk_A"));
        assert!(!is_structural_noise("c_i"));
        assert!(!is_structural_noise("SGX"));
    }

    #[test]
    fn drops_role_only_actors() {
        assert!(is_structural_noise("Adversary"));
        assert!(is_structural_noise("Adversary (A)"));
        assert!(is_structural_noise("Challenger"));
        assert!(is_structural_noise("Verifier (V)"));
        assert!(is_structural_noise("Prover"));
        assert!(is_structural_noise("Simulator"));
        // A role name with a proper name attached survives the filter.
        assert!(!is_structural_noise("Adversary Eve"));
        assert!(!is_structural_noise("Server Alice"));
    }

    #[test]
    fn keeps_real_entities() {
        assert!(!is_structural_noise("Stained Glass Transform"));
        assert!(!is_structural_noise("Intel SGX"));
        assert!(!is_structural_noise("Opal"));
        assert!(!is_structural_noise("arxiv"));
        assert!(!is_structural_noise("AES-GCM"));
        assert!(!is_structural_noise("TLS 1.3"));
        // Named tables / figures with a descriptive name survive because
        // they're not pure "Table N"-shaped strings.
        assert!(!is_structural_noise("Table of Contents"));
        assert!(!is_structural_noise("Figure skating"));
    }

    #[test]
    fn parse_drops_structural_entities_and_their_edges() {
        let parser = TupleParser::new();
        let response = r#"entity<|#|>Intel SGX<|#|>TECHNOLOGY<|#|>A real entity.
entity<|#|>Table 1<|#|>DOCUMENT<|#|>A structural reference.
entity<|#|>Adversary (A)<|#|>Other<|#|>A role-only actor.
entity<|#|>S<|#|>Other<|#|>A single-letter variable.
entity<|#|>TLS 1.3<|#|>TECHNOLOGY<|#|>Another real entity.
relation<|#|>Intel SGX<|#|>TLS 1.3<|#|>uses<|#|>Real edge.
relation<|#|>Intel SGX<|#|>Table 1<|#|>described-in<|#|>Edge into structural noise.
relation<|#|>Adversary (A)<|#|>Intel SGX<|#|>attacks<|#|>Edge out of role actor.
<|COMPLETE|>"#;

        let result = parser.parse(response, "chunk-x").unwrap();

        assert_eq!(
            result.entities.len(),
            2,
            "only real entities should survive"
        );
        assert_eq!(
            result.relationships.len(),
            1,
            "only edges between real entities survive"
        );

        let dropped_e = result
            .metadata
            .get("dropped_structural_entities")
            .and_then(|v| v.as_u64())
            .unwrap();
        assert_eq!(dropped_e, 3);

        let dropped_r = result
            .metadata
            .get("dropped_structural_relationships")
            .and_then(|v| v.as_u64())
            .unwrap();
        assert_eq!(dropped_r, 2);
    }
}
