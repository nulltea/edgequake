//! SOTA Prompt Templates for Entity Extraction
//!
//! This module contains production-quality prompts ported from LightRAG,
//! implementing tuple-based extraction format for robustness.
//!
//! ## Key Features
//!
//! - **Tuple Format**: Uses `<|#|>` delimiter for robust parsing
//! - **Completion Signal**: `<|COMPLETE|>` for reliable extraction detection
//! - **Multi-Language**: Configurable `{language}` parameter
//! - **N-ary Decomposition**: Explicit instructions for complex relationships
//! - **Entity Naming**: Title case with consistent naming rules
//!
//! ## Usage
//!
//! ```rust,ignore
//! use edgequake_pipeline::prompts::{EntityExtractionPrompts, TupleParser};
//!
//! let prompts = EntityExtractionPrompts::default();
//! let system_prompt = prompts.system_prompt(&["PERSON", "ORGANIZATION"], "English");
//! let user_prompt = prompts.user_prompt("Some text...", &["PERSON"], "English");
//!
//! // Parse LLM response
//! let parser = TupleParser::new();
//! let result = parser.parse(&llm_response, "chunk-1")?;
//! ```

mod entity_extraction;
mod normalizer;
mod parser;
mod summarization;

pub use entity_extraction::EntityExtractionPrompts;
pub use normalizer::normalize_entity_name;
pub use parser::{HybridExtractionParser, JsonExtractionParser, TupleParser};
pub use summarization::SummarizationPrompts;

/// Default tuple delimiter for extraction output.
pub const DEFAULT_TUPLE_DELIMITER: &str = "<|#|>";

/// Completion signal to detect complete extractions.
pub const DEFAULT_COMPLETION_DELIMITER: &str = "<|COMPLETE|>";

/// Supported output languages for extraction.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "Spanish",
    "French",
    "German",
    "Portuguese",
    "Italian",
    "Russian",
];

/// Default entity types for extraction.
///
/// Narrowed from the original 9-type list to 5 types tailored to the
/// research-paper RAG use case:
///
/// - Dropped `DOCUMENT` — it explicitly encouraged `Table 1`, `Section 4`,
///   `Theorem 2.1`-style structural references that polluted the graph.
/// - Dropped `PERSON` — author lists dominated real uploads (46% of all
///   entities on one sample paper). Paper-level authorship is already
///   captured in the document metadata / citation filename; per-mention
///   person entities are noise at query time.
/// - Dropped `LOCATION` and `EVENT` — low signal for the target domain
///   (cryptography / ML research); rarely appear as named things and
///   frequently surface as noise when they do.
/// - Dropped `DATE` — bare years and publication timestamps are
///   captured in document metadata (`processed_at`, front-matter year)
///   and are not useful as standalone graph entities.
///
/// `Other` is retained as the borderline-fallback bucket so the LLM has
/// somewhere to put items it can't confidently classify, without being
/// forced to mis-type them as one of the kept categories.
///
/// The complementary shape-based filter in `parser::tuple_parser` catches
/// residual noise the prompt doesn't prevent (single-letter names,
/// protocol-role actors, numbered formal labels).
pub fn default_entity_types() -> Vec<String> {
    vec![
        "ORGANIZATION".to_string(),
        "CONCEPT".to_string(),
        "TECHNOLOGY".to_string(),
        "PRODUCT".to_string(),
        "Other".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_entity_types() {
        let types = default_entity_types();
        assert!(types.contains(&"ORGANIZATION".to_string()));
        assert!(types.contains(&"CONCEPT".to_string()));
        assert!(types.contains(&"Other".to_string()));
        // Intentionally absent after the noise-reduction pass:
        assert!(!types.contains(&"PERSON".to_string()));
        assert!(!types.contains(&"LOCATION".to_string()));
        assert!(!types.contains(&"EVENT".to_string()));
        assert!(!types.contains(&"DOCUMENT".to_string()));
        assert!(!types.contains(&"DATE".to_string()));
    }

    #[test]
    fn test_supported_languages() {
        assert!(SUPPORTED_LANGUAGES.contains(&"English"));
        assert!(SUPPORTED_LANGUAGES.contains(&"Chinese"));
    }
}
