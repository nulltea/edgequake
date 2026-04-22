//! Algorithm extraction extension for EdgeQuake.
//!
//! Extracts structured algorithm definitions from documents using a 3-pass LLM pipeline:
//! 1. **Inventory** — identify algorithms/protocols/schemes in the document
//! 2. **Extraction** — produce detailed, self-contained algorithm definitions
//! 3. **Verification** — quality-check completeness and implementability
//!
//! Ported from RAGSearcher's algorithm extraction feature, adapted for EdgeQuake's
//! multi-tenant architecture and LLM provider system.

pub mod extractor;
pub mod prompts;
#[cfg(feature = "postgres")]
pub mod storage;
pub mod types;

pub use extractor::AlgorithmExtractor;
#[cfg(feature = "postgres")]
pub use storage::{
    score_algorithm_match, tokenize_algorithm_query, AlgorithmStorage, PostgresAlgorithmStorage,
};
pub use types::*;
