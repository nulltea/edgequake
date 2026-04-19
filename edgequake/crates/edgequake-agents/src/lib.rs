//! Agentic components for EdgeQuake.
//!
//! Houses features that combine external services (web search, crawl, git, LLM
//! tool-loops) to augment the core RAG pipeline. Current surface:
//!
//! - [`web_search`] — SearXNG + Crawl4AI + LLM clients used by the Reference
//!   Code GraphRAG extension (Phase 0 Layer B: find a paper's GitHub repo when
//!   PDF-native detection returns nothing).
//! - [`repo_detection`] — end-to-end orchestrator (Layer A + Layer B), domain
//!   types, and storage for `document_repos`.
//! - [`code_analysis`] — Phase 1: HTTP client for the `code-analyzer` sidecar,
//!   snippet extraction from the shared volume, and storage for
//!   `code_artifacts` + `code_reference_runs`.

pub mod code_analysis;
pub mod reference_codebase;
pub mod repo_detection;
pub mod web_search;
