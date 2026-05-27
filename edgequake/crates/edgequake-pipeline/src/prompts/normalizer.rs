//! Entity name normalization utilities.
//!
//! Provides consistent entity naming across extractions to ensure
//! proper graph node merging.
//!
//! # WHY Normalization Matters
//!
//! Without normalization, the same entity extracted from different chunks might
//! be stored as separate nodes in the knowledge graph:
//!
//! - "John Doe" (from chunk 1)
//! - "john doe" (from chunk 2)  
//! - "JOHN DOE" (from chunk 3)
//! - "The John Doe" (from chunk 4)
//!
//! This leads to:
//! 1. **Graph fragmentation**: Same entity exists as multiple disconnected nodes
//! 2. **Lost relationships**: Edges only connect to one variant
//! 3. **Query failures**: Search for "John Doe" misses "JOHN DOE" nodes
//! 4. **Inflated entity counts**: 4 nodes instead of 1
//!
//! By normalizing to `JOHN_DOE`, all references merge into a single node,
//! preserving the complete relationship graph.

/// Normalize entity name to consistent format.
///
/// Applies the following transformations:
/// - Trims and collapses whitespace
/// - Removes common prefixes (The, A, An)
/// - Removes possessive suffixes ('s)
/// - Per-word: **preserves verbatim** when the word contains any uppercase
///   letter past position 0 (acronyms like `FFN`, `TMFA`; CamelCase /
///   Pascal-with-acronym-prefix like `ISA-HiddenState`, `iPhone`,
///   `GraphQL`, `IPv6`). Otherwise title-cases (`john` → `John`).
/// - Joins words with a **single space** (not underscore) so names stay
///   readable.
///
/// This keeps acronyms intact while still merging case variants of plain
/// words (`john doe`, `John Doe` → `John Doe`). Words the model writes
/// deliberately ALL-CAPS or in `mCdonald`-style mixed case are taken at
/// face value — they're rare in well-formed Claude output and preserving
/// them is the right call for proper-noun fidelity.
///
/// # Examples
///
/// ```rust
/// use edgequake_pipeline::prompts::normalize_entity_name;
///
/// assert_eq!(normalize_entity_name("FFN"), "FFN");
/// assert_eq!(normalize_entity_name("TMFA"), "TMFA");
/// assert_eq!(normalize_entity_name("ISA-HiddenState"), "ISA-HiddenState");
/// assert_eq!(normalize_entity_name("john doe"), "John Doe");
/// assert_eq!(normalize_entity_name("the company"), "Company");
/// assert_eq!(normalize_entity_name("  Sarah  Chen  "), "Sarah Chen");
/// ```
pub fn normalize_entity_name(raw_name: &str) -> String {
    let trimmed = raw_name.trim();

    // Remove common prefixes that don't add identity
    let without_prefix = trimmed
        .strip_prefix("The ")
        .or_else(|| trimmed.strip_prefix("the "))
        .or_else(|| trimmed.strip_prefix("A "))
        .or_else(|| trimmed.strip_prefix("a "))
        .or_else(|| trimmed.strip_prefix("An "))
        .or_else(|| trimmed.strip_prefix("an "))
        .unwrap_or(trimmed);

    // Split by whitespace, normalize each word (removing possessives),
    // and rejoin with a single space.
    without_prefix
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .map(|word| {
            // Remove possessive suffix
            let without_possessive = word
                .strip_suffix("'s")
                .or_else(|| word.strip_suffix("'s"))
                .unwrap_or(word);
            preserve_or_title_case(without_possessive)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Per-word casing rule.
///
/// Preserve the word verbatim when **any** character past position 0 is
/// uppercase — this catches all-caps acronyms (`FFN`, `TMFA`, `AI`) AND
/// internal-cap words (`iPhone`, `GraphQL`, `ISA-HiddenState`,
/// `mCdonald`). Otherwise apply title case (first letter upper, rest
/// lower).
fn preserve_or_title_case(word: &str) -> String {
    let has_internal_upper = word.chars().skip(1).any(|c| c.is_uppercase());
    if has_internal_upper {
        return word.to_string();
    }
    let mut chars = word.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first
            .to_uppercase()
            .chain(chars.flat_map(|c| c.to_lowercase()))
            .collect(),
    }
}

/// Normalize for comparison (more lenient than storage normalization).
#[allow(dead_code)]
pub fn normalize_for_comparison(name: &str) -> String {
    name.trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Check if two entity names are equivalent after normalization.
#[allow(dead_code)]
pub fn entities_match(name1: &str, name2: &str) -> bool {
    normalize_entity_name(name1) == normalize_entity_name(name2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_normalization() {
        // Plain lowercase or mixed → title-cased, joined by space.
        assert_eq!(normalize_entity_name("John Doe"), "John Doe");
        assert_eq!(normalize_entity_name("john doe"), "John Doe");
    }

    #[test]
    fn test_acronyms_preserved_verbatim() {
        // Any uppercase past position 0 → preserve verbatim.
        assert_eq!(normalize_entity_name("FFN"), "FFN");
        assert_eq!(normalize_entity_name("TMFA"), "TMFA");
        assert_eq!(normalize_entity_name("AI"), "AI");
        assert_eq!(normalize_entity_name("ML"), "ML");
        assert_eq!(normalize_entity_name("JOHN DOE"), "JOHN DOE");
    }

    #[test]
    fn test_camelcase_and_internal_caps_preserved() {
        assert_eq!(normalize_entity_name("ISA-HiddenState"), "ISA-HiddenState");
        assert_eq!(normalize_entity_name("iPhone"), "iPhone");
        assert_eq!(normalize_entity_name("eBay"), "eBay");
        assert_eq!(normalize_entity_name("GraphQL"), "GraphQL");
        assert_eq!(normalize_entity_name("OAuth"), "OAuth");
        assert_eq!(normalize_entity_name("IPv6"), "IPv6");
    }

    #[test]
    fn test_whitespace_handling() {
        // Collapse whitespace to single space.
        assert_eq!(normalize_entity_name("  John  Doe  "), "John Doe");
        assert_eq!(normalize_entity_name("\tJohn\nDoe\r"), "John Doe");
        assert_eq!(normalize_entity_name("John   Doe"), "John Doe");
    }

    #[test]
    fn test_prefix_removal() {
        assert_eq!(normalize_entity_name("The Company"), "Company");
        assert_eq!(normalize_entity_name("the company"), "Company");
        assert_eq!(normalize_entity_name("A Person"), "Person");
        assert_eq!(normalize_entity_name("An Event"), "Event");
        // Acronym kept after prefix strip.
        assert_eq!(normalize_entity_name("The FFN"), "FFN");
    }

    #[test]
    fn test_possessive_removal() {
        assert_eq!(normalize_entity_name("John's"), "John");
        assert_eq!(
            normalize_entity_name("Company's Products"),
            "Company Products"
        );
    }

    #[test]
    fn test_mixed_examples_from_real_docs() {
        // User-reported regressions from lattice extractions.
        assert_eq!(
            normalize_entity_name("Vocab Memorization"),
            "Vocab Memorization"
        );
        assert_eq!(
            normalize_entity_name("Wire-side Leakage"),
            "Wire-side Leakage"
        );
    }

    #[test]
    fn test_empty_and_edge_cases() {
        assert_eq!(normalize_entity_name(""), "");
        assert_eq!(normalize_entity_name("   "), "");
        assert_eq!(normalize_entity_name("A"), "A");
        assert_eq!(normalize_entity_name("I"), "I");
    }

    #[test]
    fn test_entities_match() {
        assert!(entities_match("John Doe", "john doe"));
        assert!(entities_match("The Company", "Company"));
        assert!(entities_match("  Sarah  ", "Sarah"));
        assert!(!entities_match("John", "Jane"));
    }

    #[test]
    fn test_normalize_for_comparison() {
        assert_eq!(normalize_for_comparison("  John  Doe  "), "john doe");
        assert_eq!(normalize_for_comparison("JOHN DOE"), "john doe");
    }

    #[test]
    fn test_special_characters_preserved() {
        // Hyphens are NOT word separators; "New-York" is one whitespace token.
        // The "Y" at position 4 is uppercase past position 0, so the word is
        // treated as internal-cap and preserved verbatim.
        assert_eq!(normalize_entity_name("New-York"), "New-York");
        // All-lowercase hyphenated word stays one token, gets first-letter
        // title-cased only.
        assert_eq!(normalize_entity_name("new-york"), "New-york");
        assert_eq!(normalize_entity_name("C++"), "C++");
    }
}
