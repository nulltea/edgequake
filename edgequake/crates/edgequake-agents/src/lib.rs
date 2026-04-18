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

pub mod repo_detection;
pub mod web_search;
