//! Web-search fallback for paper → reference-repo resolution.
//!
//! Layer B of the Phase 0 detection flow. Invoked when direct PDF-link
//! extraction ([`edgequake_pdf::extract_links`]) returns nothing useful.
//! Pipeline:
//!
//! 1. [`front_matter`] — extract title + first-author from the paper markdown.
//! 2. [`github`] — query the GitHub Search API for repositories matching
//!    the paper. Authenticated with `GITHUB_TOKEN` for proper rate limits.
//! 3. [`resolver`] — pick the best hit (or hand the LLM a shortlist when
//!    the top one isn't obviously right), then validate with
//!    [`edgequake_pdf::repos`].
//!
//! The legacy SearXNG + Crawl4AI path is kept compiled (still used by the
//! verifier to fetch READMEs for non-GitHub hosts) but no longer drives
//! repo discovery — upstream search engines kept tripping over CAPTCHAs.

pub mod crawl4ai;
pub mod front_matter;
pub mod github;
pub mod resolver;
pub mod searxng;

pub use crawl4ai::{Crawl4aiClient, Crawl4aiError, MarkdownResponse};
pub use front_matter::{extract_front_matter, PaperFrontMatter};
pub use github::{build_repo_query, GithubClient, GithubError, GithubSearchHit};
pub use resolver::{
    resolve_repo, HitSource, RepoResolverConfig, RepoResolverError, ResolvedRepo,
};
pub use searxng::{SearxngClient, SearxngError, SearxngResult};
