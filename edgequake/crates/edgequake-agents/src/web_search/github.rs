//! GitHub Search API client for Layer B repo discovery.
//!
//! Replaces the SearXNG-based path that kept getting rate-limited or
//! CAPTCHA-blocked by upstream search engines (Brave, DDG, Startpage).
//! Going straight to `api.github.com/search/repositories` is purpose-built
//! for "find a repo for paper X" — we just feed it title + author tokens
//! and rank by GitHub's own relevance + star count.
//!
//! Falls back to unauthenticated requests when `GITHUB_TOKEN` is unset.
//! That mode has a 60-req/hr search-API limit which is fine for dev but
//! will throttle quickly in production — set a PAT.

use std::sync::Arc;

use base64::Engine;
use octocrab::Octocrab;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

#[derive(Debug, Error)]
pub enum GithubError {
    #[error("octocrab error: {0}")]
    Octocrab(String),
    #[error("readme decode failed: {0}")]
    ReadmeDecode(String),
}

impl From<octocrab::Error> for GithubError {
    fn from(e: octocrab::Error) -> Self {
        GithubError::Octocrab(e.to_string())
    }
}

/// Slim repository record extracted from GitHub Search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GithubSearchHit {
    pub owner: String,
    pub repo: String,
    pub html_url: String,
    pub description: Option<String>,
    pub stargazers_count: u64,
    /// 0-based rank in the search response (sorted by GitHub relevance).
    pub rank: usize,
}

/// Thin wrapper around `octocrab::Octocrab`. Cheap to clone — wraps an
/// `Arc` internally — so we can hand copies into the orchestrator + verifier
/// without holding a lock.
#[derive(Clone)]
pub struct GithubClient {
    inner: Arc<Octocrab>,
    /// Whether we authenticated (affects rate-limit messaging only).
    pub authenticated: bool,
}

impl GithubClient {
    /// Build a client from `$GITHUB_TOKEN`. Falls back to anonymous mode
    /// when the env var is unset or empty.
    pub fn from_env() -> Self {
        let token = std::env::var("GITHUB_TOKEN")
            .ok()
            .filter(|s| !s.trim().is_empty());
        Self::new(token.as_deref())
    }

    pub fn new(token: Option<&str>) -> Self {
        // rustls 0.23 requires an explicit crypto provider when multiple
        // backends (`aws-lc-rs` + `ring`) are in the dep graph — which is
        // our case (octocrab pulls `ring`, reqwest pulls `aws-lc-rs`).
        // Without this call, the first connector build panics with
        // "Could not automatically determine the process-level
        // CryptoProvider". Use `OnceLock` to make it safe to call from
        // every constructor; installation is idempotent (Err on
        // already-installed is dropped).
        static CRYPTO: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        CRYPTO.get_or_init(|| {
            let _ = rustls::crypto::ring::default_provider().install_default();
        });

        // Octocrab's typestate builder requires the auth state to be set
        // before `.build()` is available — so we use the typed builder
        // for authenticated clients and `Octocrab::default()` for the
        // unauthenticated path (default == NoAuth, no API key needed).
        let (inner, authenticated) = match token {
            Some(t) if !t.is_empty() => {
                let built = Octocrab::builder()
                    .personal_token(t.to_string())
                    .build()
                    .expect("octocrab personal-token build must succeed");
                (built, true)
            }
            _ => (Octocrab::default(), false),
        };
        Self {
            inner: Arc::new(inner),
            authenticated,
        }
    }

    /// Search GitHub for repositories matching `query`. Returns up to
    /// `limit` hits, ranked by GitHub's relevance + star count.
    pub async fn search_repos(
        &self,
        query: &str,
        limit: u8,
    ) -> Result<Vec<GithubSearchHit>, GithubError> {
        let page = self
            .inner
            .search()
            .repositories(query)
            .sort("best-match")
            .per_page(limit.clamp(1, 100))
            .send()
            .await?;

        let hits: Vec<GithubSearchHit> = page
            .items
            .into_iter()
            .enumerate()
            .filter_map(|(rank, repo)| {
                let owner = repo.owner.as_ref()?.login.clone();
                let name = repo.name.clone();
                let html_url = repo
                    .html_url
                    .as_ref()
                    .map(|u| u.to_string())
                    .unwrap_or_else(|| format!("https://github.com/{owner}/{name}"));
                Some(GithubSearchHit {
                    owner,
                    repo: name,
                    html_url,
                    description: repo.description.clone(),
                    stargazers_count: repo.stargazers_count.unwrap_or(0) as u64,
                    rank,
                })
            })
            .collect();
        debug!(query = %query, hit_count = hits.len(), "github search returned");
        Ok(hits)
    }

    /// Fetch and decode the default-branch README. Returns `Ok(None)` when
    /// the repo has no README; surfaces other errors so the caller can
    /// decide between falling back to Crawl4AI or treating the candidate
    /// as unverifiable.
    pub async fn fetch_readme(
        &self,
        owner: &str,
        repo: &str,
    ) -> Result<Option<String>, GithubError> {
        let result = self.inner.repos(owner, repo).get_readme().send().await;
        let content = match result {
            Ok(c) => c,
            Err(octocrab::Error::GitHub { source, .. }) if source.status_code == 404 => {
                debug!(owner, repo, "github readme: 404");
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };
        // Octocrab's `Content` exposes the base64-encoded `content`
        // field directly. Decode it ourselves rather than call the
        // raw-fetch endpoint — one request, no second auth hit.
        let Some(b64) = content.content else {
            return Ok(None);
        };
        // GitHub wraps base64 content at 60 chars per line; strip
        // whitespace before decoding.
        let stripped: String = b64.chars().filter(|c| !c.is_whitespace()).collect();
        match base64::engine::general_purpose::STANDARD.decode(stripped.as_bytes()) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(s) => Ok(Some(s)),
                Err(e) => {
                    warn!(owner, repo, error = %e, "github readme: not valid utf-8");
                    Err(GithubError::ReadmeDecode(e.to_string()))
                }
            },
            Err(e) => Err(GithubError::ReadmeDecode(e.to_string())),
        }
    }
}

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "the", "of", "for", "to", "in", "on", "with", "by", "is", "are", "at",
    "from", "via", "using", "as", "or",
];

/// Build an ordered list of GitHub Search queries for one paper. Caller
/// runs them all and dedupes hits by `{owner}/{repo}`. Generating
/// multiple queries — instead of one finely-tuned one — sidesteps three
/// GitHub-Search quirks:
///
/// 1. **Forks are excluded by default.** Academic repos are often forked
///    (e.g. authors' working fork lives at `wuwuz/Pacmann`, original
///    elsewhere). Every query appends `fork:true` to surface those.
/// 2. **Keyword queries are AND-style across one field.** A 7-token
///    title + `in:name,description,readme` requires *all* tokens to
///    appear in the same indexed field — almost no real repo passes
///    that bar, while indices like `awesome-*` lists do.
/// 3. **Acronym-only queries are too broad.** `PACMANN fork:true`
///    returns 945 results dominated by hobby projects ("daniken/Pacmann")
///    that bury the paper's repo. Pairing the acronym with *any*
///    distinctive title token cuts the result set to 3–10 hits with the
///    right repo at rank 0.
///
/// Strategy:
/// - `{ACRONYM} {title-token} fork:true` for each of the top
///   title-core tokens (max 4). Empirically these surface niche
///   academic repos that broader queries miss.
/// - `{ACRONYM} {surname} fork:true` — narrows the acronym hits when
///   the surname is in the README/description.
/// - `{ACRONYM} fork:true` — broad fallback if everything above misses.
/// - `{title-core} {surname} in:name,description,readme fork:true` —
///   the only query for papers without an acronym, otherwise the
///   final fallback.
pub fn build_repo_queries(title: &str, first_author: Option<&str>) -> Vec<String> {
    let mut queries: Vec<String> = Vec::new();
    let surname = first_author
        .and_then(|a| a.split_whitespace().last())
        .filter(|s| s.len() >= 3);

    let acronym = extract_acronym(title);
    // Filter out the acronym itself from the token list so we don't
    // emit `ACR ACR fork:true`.
    let title_tokens: Vec<String> = title_tokens(title, 6)
        .into_iter()
        .filter(|t| {
            acronym
                .as_ref()
                .map(|a| !t.eq_ignore_ascii_case(a))
                .unwrap_or(true)
        })
        .collect();

    if let Some(acr) = &acronym {
        // Pair the acronym with each meaningful title token. Cap at 4
        // to keep the worst-case query count predictable.
        for token in title_tokens.iter().take(4) {
            queries.push(format!("{acr} {token} fork:true"));
        }
        if let Some(s) = surname {
            queries.push(format!("{acr} {s} fork:true"));
        }
        // Acronym alone — broad fallback.
        queries.push(format!("{acr} fork:true"));
    }

    // Title-core fallback. Always present so papers without an acronym
    // get a query at all, and acronym-papers get a backstop.
    let core = title_tokens.join(" ");
    if !core.is_empty() {
        let mut q = core;
        if let Some(s) = surname {
            q.push(' ');
            q.push_str(s);
        }
        q.push_str(" in:name,description,readme fork:true");
        queries.push(q);
    }

    queries
}

/// Backwards-compat single-query helper. Returns the first (most
/// precise) query from [`build_repo_queries`].
pub fn build_repo_query(title: &str, first_author: Option<&str>) -> String {
    build_repo_queries(title, first_author)
        .into_iter()
        .next()
        .unwrap_or_else(|| format!("{title} fork:true"))
}

/// Find an all-caps acronym in the title, e.g. "PACMANN: Efficient ..."
/// → `Some("PACMANN")`. Heuristic: pick the first standalone token of
/// length ≥3 whose alphabetic characters are all uppercase. Returns
/// `None` if no token qualifies (titles like "Stained Glass Transform"
/// fall through to the broader strategy).
fn extract_acronym(title: &str) -> Option<String> {
    title
        .split(|c: char| !c.is_alphanumeric())
        .find(|t| {
            t.chars().filter(|c| c.is_alphabetic()).count() >= 3
                && t.chars().filter(|c| c.is_alphabetic()).all(|c| c.is_uppercase())
        })
        .map(|s| s.to_string())
}

/// Tokenise the title into content-bearing words. Drops stopwords and
/// single-char fragments. Returns owned strings so the caller can stitch
/// them into queries independently of the original `title`'s lifetime.
fn title_tokens(title: &str, max_tokens: usize) -> Vec<String> {
    title
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|t| t.len() > 1)
        .filter(|t| !STOPWORDS.contains(&t.to_ascii_lowercase().as_str()))
        .take(max_tokens)
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_pair_acronym_with_each_title_token() {
        let qs = build_repo_queries(
            "PACMANN: Efficient Private Approximate Nearest Neighbor Search",
            Some("Mingxun Zhou"),
        );
        // First few queries pair the acronym with each title token —
        // these are the ones that surface niche academic repos at rank 0.
        assert!(qs.iter().any(|q| q == "PACMANN Efficient fork:true"));
        assert!(qs.iter().any(|q| q == "PACMANN Private fork:true"));
        assert!(qs.iter().any(|q| q == "PACMANN Approximate fork:true"));
        // Surname-augmented query.
        assert!(qs.iter().any(|q| q == "PACMANN Zhou fork:true"));
        // Broad acronym-only fallback.
        assert!(qs.iter().any(|q| q == "PACMANN fork:true"));
        // Title-core fallback keeps the readme qualifier.
        assert!(qs
            .iter()
            .any(|q| q.contains("in:name,description,readme") && q.contains("Zhou")));
        // Acronym never paired with itself.
        for q in &qs {
            assert!(!q.contains("PACMANN PACMANN"), "duplicated acronym: {q}");
        }
    }

    #[test]
    fn queries_skip_acronym_when_title_has_none() {
        let qs = build_repo_queries("Stained Glass Transform Embeddings", Some("Alice Smith"));
        // No all-caps token → only the fallback query.
        assert_eq!(qs.len(), 1, "expected 1 query, got {:?}", qs);
        assert!(qs[0].contains("Stained"));
        assert!(qs[0].contains("Smith"));
        assert!(qs[0].contains("in:name,description,readme"));
        assert!(qs[0].contains("fork:true"));
    }

    #[test]
    fn queries_skip_author_branch_when_missing() {
        let qs = build_repo_queries("BERT: A New Language Model", None);
        // Acronym pair queries are present, surname queries aren't.
        assert!(qs.iter().any(|q| q == "BERT New fork:true"));
        assert!(!qs.iter().any(|q| q.contains("Smith") || q.contains("Devlin")));
    }

    #[test]
    fn single_query_helper_returns_first() {
        // First query is `{acronym} {first-title-token} fork:true`.
        let q = build_repo_query("BERT Language Model", Some("Devlin"));
        assert_eq!(q, "BERT Language fork:true");
    }

    #[test]
    fn extract_acronym_finds_all_caps_token() {
        assert_eq!(extract_acronym("PACMANN: foo"), Some("PACMANN".into()));
        assert_eq!(extract_acronym("BERT for NLP"), Some("BERT".into()));
        // First all-caps wins.
        assert_eq!(extract_acronym("xyz ABC DEF"), Some("ABC".into()));
        // Length-2 doesn't qualify (too noisy — would match "AN", "OR", etc.).
        assert_eq!(extract_acronym("AI for ML systems"), None);
        // Plain title has none.
        assert_eq!(extract_acronym("a study of foo"), None);
    }

    // Construction tests need a Tokio runtime — octocrab's tower buffer
    // spawns an internal worker task, which panics in a sync test.
    #[tokio::test]
    async fn github_client_builds_without_token() {
        let c = GithubClient::new(None);
        assert!(!c.authenticated);
    }

    #[tokio::test]
    async fn github_client_builds_with_token() {
        let c = GithubClient::new(Some("ghp_dummy"));
        assert!(c.authenticated);
    }
}
