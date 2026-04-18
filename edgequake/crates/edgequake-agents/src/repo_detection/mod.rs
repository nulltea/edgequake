//! Reference-repository detection for the Reference Code GraphRAG extension.
//!
//! Public surface:
//! - [`orchestrator::run_detection`] — runs Layer A (PDF annotations) then, if
//!   nothing was found, Layer B (web search via [`crate::web_search`]).
//! - [`types`] — domain types mirroring `migrations/041_add_document_repos.sql`.
//! - [`storage`] — trait + Postgres implementation for persisting candidates
//!   and tracking per-document detection state.
//!
//! The API-layer `TaskProcessor` wraps these: it pulls the document's PDF
//! bytes + markdown, runs `run_detection`, and persists via [`storage`].

pub mod orchestrator;
pub mod storage;
pub mod types;

pub use orchestrator::{
    run_detection, DetectionOutcome, RepoDetectionConfig, RepoDetectionError, WebSearchClients,
};
#[cfg(feature = "postgres")]
pub use storage::PostgresRepoStorage;
pub use storage::{RepoStorage, RepoStorageError};
pub use types::{
    Confidence, DetectionMethod, DetectionRun, DetectionRunStatus, DocumentRepo, RepoCandidate,
    RepoHost, RepoStatus,
};
