//! Web-search fallback for paper → reference-repo resolution.
//!
//! Layer B of the Phase 0 detection flow (plan §4/Phase 0). Invoked when
//! direct PDF-link extraction ([`edgequake_pdf::extract_links`]) returns no
//! candidate repositories. Pipeline:
//!
//! 1. [`front_matter`] — extract title + first-author from the paper markdown.
//! 2. [`searxng`] — issue a search for `"{first_author} {title} github"`.
//! 3. [`crawl4ai`] — fetch cleaned markdown for the top N results.
//! 4. [`resolver`] — ask the workspace LLM to pick one canonical
//!    `{host}/{owner}/{repo}`, then validate with [`edgequake_pdf::repos`].

pub mod crawl4ai;
pub mod front_matter;
pub mod resolver;
pub mod searxng;

pub use crawl4ai::{Crawl4aiClient, Crawl4aiError, MarkdownResponse};
pub use front_matter::{extract_front_matter, PaperFrontMatter};
pub use resolver::{resolve_repo, RepoResolverConfig, RepoResolverError, ResolvedRepo};
pub use searxng::{SearxngClient, SearxngError, SearxngResult};
