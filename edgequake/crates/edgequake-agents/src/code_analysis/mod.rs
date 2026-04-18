//! Reference-code analysis (Phase 1 of the Reference Code GraphRAG extension).
//!
//! Given an approved `document_repos` row and the document's approved
//! `algorithms`, call the `code-analyzer` sidecar to localize each algorithm
//! in the repo, read the matched snippet from the shared volume, and persist
//! it as a `code_artifacts` candidate for human review.
//!
//! The sidecar drives Claude Code headless; this module is the edgequake-side
//! glue (types, HTTP client, snippet reader, storage).

pub mod client;
pub mod snippet;
pub mod storage;
pub mod types;

pub use client::{AnalyzerClient, AnalyzerClientError};
pub use snippet::{extract as extract_snippet, language_from_path, ExtractedSnippet, SnippetError};
#[cfg(feature = "postgres")]
pub use storage::PostgresCodeArtifactStorage;
pub use storage::{CodeArtifactStorage, CodeStorageError};
pub use types::{
    AnalyzerAlgorithmInput, AnalyzerFinding, AnalyzerRequest, AnalyzerResponse, ArtifactStatus,
    CodeArtifact, CodeArtifactCandidate, CodeReferenceRun, MatchConfidence, RunStatus,
};
