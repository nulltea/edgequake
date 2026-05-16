//! Orchestrator: paper markdown → candidate reference repository URL.
//!
//! Two discovery paths run in **parallel** then combine + dedup before a
//! single LLM tie-breaker decides:
//!
//! 1. **Layer B — GitHub Search API** ([`gather_github_hits`]). Multiple
//!    targeted queries through octocrab; primary path because GitHub's
//!    repo index has the most relevant signal for "what repo implements
//!    this paper."
//! 2. **Layer C — SearXNG + Crawl4AI** ([`gather_web_search_hits`]).
//!    A `site:github.com` meta-search through SearXNG, followed by
//!    Crawl4AI fetches on the top results to surface embedded
//!    `github.com/{owner}/{repo}` URLs. Catches niche academic repos
//!    whose GitHub Search ranking is poor but whose paper page on
//!    arXiv / Semantic Scholar / OpenReview links straight to the repo.
//!
//! After both finish:
//! - Hits are merged and deduped by `(owner, repo)`. Source attribution
//!   prefers whichever path saw the repo first.
//! - The top N (≤8) hits have their READMEs fetched via the GitHub API
//!   (same endpoint regardless of which path surfaced them) so the LLM
//!   sees consistent context.
//! - One LLM call picks the best repo. `NONE` → `NotFound` (we don't
//!   silently substitute the top hit; the workspace filter should see a
//!   clean miss instead of a wrong answer).

use std::sync::Arc;

use edgequake_llm::traits::LLMProvider;
use edgequake_pdf::{DetectedRepo, RepoHost};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

use super::crawl4ai::Crawl4aiClient;
use super::front_matter::extract_front_matter;
use super::github::{build_repo_queries, GithubClient, GithubError, GithubSearchHit};
use super::searxng::SearxngClient;
use super::{Crawl4aiError, SearxngError};

#[derive(Debug, Error)]
pub enum RepoResolverError {
    #[error("front-matter extraction failed: could not find a title in the paper markdown")]
    NoFrontMatter,
    #[error(transparent)]
    Searxng(#[from] SearxngError),
    #[error(transparent)]
    Crawl4ai(#[from] Crawl4aiError),
    #[error(transparent)]
    Github(#[from] GithubError),
    #[error("LLM call failed: {0}")]
    Llm(String),
    #[error("resolver found no repository candidates")]
    NotFound,
}

#[derive(Debug, Clone)]
pub struct RepoResolverConfig {
    /// Max GitHub Search hits per query.
    pub max_github_hits: u8,
    /// Max SearXNG result pages to fetch via Crawl4AI when scanning for
    /// embedded repo URLs. Each page is one extra HTTP round-trip;
    /// 4 is a balance between coverage and latency.
    pub max_web_search_pages: usize,
    /// Hard cap on README chars per hit when building the LLM shortlist.
    pub chars_per_hit_for_llm: usize,
    /// Upper bound on the LLM prompt string.
    pub llm_prompt_char_budget: usize,
    /// Max hits passed to the LLM after combining + deduping.
    pub max_shortlist_size: usize,
}

impl Default for RepoResolverConfig {
    fn default() -> Self {
        Self {
            max_github_hits: 10,
            max_web_search_pages: 4,
            chars_per_hit_for_llm: 2_000,
            llm_prompt_char_budget: 12_000,
            max_shortlist_size: 8,
        }
    }
}

/// Where in the discovery pipeline a hit was found. Carried through to
/// `DetectionMethod` so the UI can tell `github_api` rows apart from
/// `web_search` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HitSource {
    GithubApi,
    WebSearch,
}

/// The output surfaced from the resolver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedRepo {
    pub host: RepoHost,
    pub owner: String,
    pub repo: String,
    pub url: String,
    /// 0-based rank within whichever source surfaced this repo.
    pub source_rank: Option<usize>,
    /// The URL we attribute the find to (the GitHub repo page, or a
    /// SearXNG result URL when Layer C found it).
    pub source_url: Option<String>,
    /// Which Layer surfaced this repo.
    pub source: HitSource,
}

impl ResolvedRepo {
    fn from_detected(r: DetectedRepo, source: HitSource) -> Self {
        Self {
            host: r.host,
            owner: r.owner,
            repo: r.repo,
            url: r.url,
            source_rank: None,
            source_url: None,
            source,
        }
    }
}

/// A hit shape both paths produce. SearXNG hits have `description`
/// unset and `stargazers_count == 0` until we enrich via GitHub later;
/// that's fine because the LLM prompt doesn't *require* those fields.
#[derive(Debug, Clone)]
struct Hit {
    inner: GithubSearchHit,
    source: HitSource,
}

/// Resolve the paper to a single canonical GitHub repository by running
/// Layer B + Layer C in parallel and letting the LLM pick from the
/// combined shortlist.
pub async fn resolve_repo(
    paper_markdown: &str,
    github: &GithubClient,
    searxng: &SearxngClient,
    crawl: &Crawl4aiClient,
    llm: Arc<dyn LLMProvider>,
    config: &RepoResolverConfig,
) -> Result<ResolvedRepo, RepoResolverError> {
    let fm = extract_front_matter(paper_markdown).ok_or(RepoResolverError::NoFrontMatter)?;

    // Layers B + C run concurrently; each path is best-effort so a
    // SearXNG outage doesn't stall the GitHub path and vice versa.
    let (gh_result, web_result) = tokio::join!(
        gather_github_hits(github, &fm, config),
        gather_web_search_hits(searxng, crawl, &fm, config),
    );

    let gh_hits = gh_result.unwrap_or_else(|e| {
        warn!(error = %e, "github gather failed; continuing with web-search hits only");
        Vec::new()
    });
    let web_hits = web_result.unwrap_or_else(|e| {
        warn!(error = %e, "web-search gather failed; continuing with github hits only");
        Vec::new()
    });

    let merged = merge_hits(gh_hits, web_hits, config.max_shortlist_size);
    if merged.is_empty() {
        return Err(RepoResolverError::NotFound);
    }
    info!(merged_count = merged.len(), "combined shortlist ready");

    // Fetch READMEs via the GitHub API regardless of source — same
    // owner/repo, same endpoint, consistent prompt context.
    let mut shortlist: Vec<(Hit, String)> = Vec::with_capacity(merged.len());
    for hit in &merged {
        let readme = github
            .fetch_readme(&hit.inner.owner, &hit.inner.repo)
            .await
            .unwrap_or_else(|e| {
                warn!(
                    owner = %hit.inner.owner,
                    repo = %hit.inner.repo,
                    error = %e,
                    "readme fetch failed"
                );
                None
            })
            .unwrap_or_default();
        shortlist.push((hit.clone(), readme));
    }

    // LLM tie-breaker over the combined set.
    let prompt = build_llm_prompt(&fm.title, fm.first_author.as_deref(), &shortlist, config);
    let resp = llm
        .complete(&prompt)
        .await
        .map_err(|e| RepoResolverError::Llm(e.to_string()))?;
    let raw = resp.content.clone();

    match parse_repo_url(raw.trim()) {
        Some(repo) => {
            let matched = shortlist.iter().find(|(h, _)| {
                h.inner.html_url.eq_ignore_ascii_case(&repo.url)
                    || (h.inner.owner.eq_ignore_ascii_case(&repo.owner)
                        && h.inner.repo.eq_ignore_ascii_case(&repo.repo))
            });
            let (source, source_rank, source_url) = match matched {
                Some((h, _)) => (
                    h.source,
                    Some(h.inner.rank),
                    Some(h.inner.html_url.clone()),
                ),
                // LLM returned a URL that doesn't appear in our shortlist;
                // attribute to GithubApi as a sensible default — most LLM
                // hallucinations end up as github.com URLs anyway.
                None => (HitSource::GithubApi, None, None),
            };
            Ok(ResolvedRepo {
                source_rank,
                source_url,
                source,
                ..ResolvedRepo::from_detected(repo, source)
            })
        }
        None => {
            warn!(raw = %raw, "LLM rejected every shortlist candidate; returning NotFound");
            Err(RepoResolverError::NotFound)
        }
    }
}

/// Layer B: pull hits from GitHub Search via the multi-query strategy
/// in [`build_repo_queries`]. Hits are deduped by `(owner, repo)` in
/// accumulation order so the first query's rank-0 wins.
async fn gather_github_hits(
    github: &GithubClient,
    fm: &super::front_matter::PaperFrontMatter,
    config: &RepoResolverConfig,
) -> Result<Vec<Hit>, RepoResolverError> {
    let queries = build_repo_queries(&fm.title, fm.first_author.as_deref());
    let mut hits: Vec<GithubSearchHit> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for query in &queries {
        info!(query = %query, authenticated = github.authenticated, "github repo search");
        match github.search_repos(query, config.max_github_hits).await {
            Ok(round) => {
                for h in round {
                    let key = format!("{}/{}", h.owner.to_ascii_lowercase(), h.repo.to_ascii_lowercase());
                    if seen.insert(key) {
                        hits.push(h);
                    }
                }
            }
            Err(e) => warn!(query = %query, error = %e, "github search query failed"),
        }
    }
    for (i, h) in hits.iter_mut().enumerate() {
        h.rank = i;
    }
    Ok(hits
        .into_iter()
        .map(|h| Hit {
            inner: h,
            source: HitSource::GithubApi,
        })
        .collect())
}

/// Layer C: meta-search via SearXNG, then Crawl4AI fetches the top
/// pages to scrape embedded `github.com/{owner}/{repo}` URLs.
///
/// SearXNG result URLs themselves are also parsed — if a result *is*
/// a github.com repo page, that's the cheapest possible hit.
async fn gather_web_search_hits(
    searxng: &SearxngClient,
    crawl: &Crawl4aiClient,
    fm: &super::front_matter::PaperFrontMatter,
    config: &RepoResolverConfig,
) -> Result<Vec<Hit>, RepoResolverError> {
    let primary_q = build_searxng_query(&fm.title, fm.first_author.as_deref());
    let github_q = build_site_github_query(&fm.title);
    debug!(primary = %primary_q, github = %github_q, "searxng queries");

    let (primary, fallback) = tokio::join!(searxng.search(&primary_q), searxng.search(&github_q));
    let primary = primary.unwrap_or_else(|e| {
        warn!(error = %e, "searxng primary query failed");
        Vec::new()
    });
    let fallback = fallback.unwrap_or_else(|e| {
        warn!(error = %e, "searxng github-scoped query failed");
        Vec::new()
    });

    // Merge: prefer the lowest rank across both queries. Github-scoped
    // results tend to be more on-target so they win ties.
    let mut by_url: std::collections::HashMap<String, (usize, super::searxng::SearxngResult)> =
        std::collections::HashMap::new();
    for (rank, r) in primary.into_iter().enumerate() {
        by_url.entry(r.url.clone()).or_insert((rank, r));
    }
    for (rank, r) in fallback.into_iter().enumerate() {
        by_url
            .entry(r.url.clone())
            .and_modify(|(existing, _)| {
                if rank < *existing {
                    *existing = rank;
                }
            })
            .or_insert((rank, r));
    }
    let mut merged: Vec<(usize, super::searxng::SearxngResult)> = by_url.into_values().collect();
    merged.sort_by_key(|(r, _)| *r);
    debug!(searxng_merged = merged.len(), "searxng merged result count");

    let mut hits: Vec<Hit> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Pass 1: SearXNG URLs that are already github repo pages.
    for (rank, r) in &merged {
        if let Some(repo) = parse_repo_url(&r.url) {
            let key = format!(
                "{}/{}",
                repo.owner.to_ascii_lowercase(),
                repo.repo.to_ascii_lowercase()
            );
            if seen.insert(key) {
                hits.push(Hit {
                    inner: GithubSearchHit {
                        owner: repo.owner.clone(),
                        repo: repo.repo.clone(),
                        html_url: repo.url.clone(),
                        description: None,
                        stargazers_count: 0,
                        rank: hits.len(),
                    },
                    source: HitSource::WebSearch,
                });
                debug!(rank = rank, url = %r.url, "searxng direct repo hit");
            }
        }
    }

    // Pass 2: Crawl4AI fetches on the top-N non-github SearXNG pages.
    // Many papers link to their repo from arxiv-html / openreview /
    // huggingface — those pages aren't github repos themselves but
    // embed the canonical repo URL in body text.
    let crawl_targets: Vec<&str> = merged
        .iter()
        .filter(|(_, r)| parse_repo_url(&r.url).is_none())
        .take(config.max_web_search_pages)
        .map(|(_, r)| r.url.as_str())
        .collect();
    for url in crawl_targets {
        match crawl.markdown(url).await {
            Ok(resp) if resp.success => {
                for repo in scan_markdown_for_repos(&resp.markdown) {
                    let key = format!(
                        "{}/{}",
                        repo.owner.to_ascii_lowercase(),
                        repo.repo.to_ascii_lowercase()
                    );
                    if seen.insert(key) {
                        hits.push(Hit {
                            inner: GithubSearchHit {
                                owner: repo.owner.clone(),
                                repo: repo.repo.clone(),
                                html_url: repo.url.clone(),
                                description: None,
                                stargazers_count: 0,
                                rank: hits.len(),
                            },
                            source: HitSource::WebSearch,
                        });
                        debug!(url = %url, repo = %repo.url, "crawl4ai embedded repo hit");
                    }
                }
            }
            Ok(_) => warn!(url = %url, "crawl4ai returned success=false"),
            Err(e) => warn!(url = %url, error = %e, "crawl4ai fetch failed"),
        }
    }

    Ok(hits)
}

/// Combine hits from Layer B and Layer C, deduping by `(owner, repo)`.
/// GitHub hits keep their order first (since the API has the better
/// relevance signal), then unique web-search hits append after. The
/// resulting slice is capped at `max` to keep the LLM prompt bounded.
fn merge_hits(gh: Vec<Hit>, web: Vec<Hit>, max: usize) -> Vec<Hit> {
    let mut out: Vec<Hit> = Vec::with_capacity(gh.len() + web.len());
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for h in gh.into_iter().chain(web.into_iter()) {
        let key = format!(
            "{}/{}",
            h.inner.owner.to_ascii_lowercase(),
            h.inner.repo.to_ascii_lowercase()
        );
        if seen.insert(key) {
            out.push(h);
            if out.len() >= max {
                break;
            }
        }
    }
    // Re-rank in merge order; per-source rank is preserved on the
    // inner struct for attribution but the LLM prompt's "#rank" uses
    // this combined view.
    for (i, h) in out.iter_mut().enumerate() {
        h.inner.rank = i;
    }
    out
}

fn build_searxng_query(title: &str, author: Option<&str>) -> String {
    match author {
        Some(a) => format!("{a} {title} github"),
        None => format!("{title} github"),
    }
}

/// `site:github.com {title-core}` — forces SearXNG to surface github
/// repo pages directly. Clips to the most content-bearing title tokens
/// so the query stays useful for verbose academic titles.
fn build_site_github_query(title: &str) -> String {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "the", "of", "for", "to", "in", "on", "with", "by", "is", "are", "at",
        "from", "via", "using", "as", "or",
    ];
    const MAX_TOKENS: usize = 8;
    let core: Vec<&str> = title
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|t| !t.is_empty())
        .filter(|t| !STOPWORDS.contains(&t.to_ascii_lowercase().as_str()))
        .take(MAX_TOKENS)
        .collect();
    if core.is_empty() {
        return format!("site:github.com {title}");
    }
    format!("site:github.com {}", core.join(" "))
}

/// Scan free-form markdown for embedded github/gitlab/bitbucket repo
/// URLs. Returns all unique hits in order of first appearance.
fn scan_markdown_for_repos(md: &str) -> Vec<DetectedRepo> {
    let mut out: Vec<DetectedRepo> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for token in md.split(|c: char| {
        c.is_whitespace() || matches!(c, '[' | ']' | '(' | ')' | '<' | '>' | '"' | '\'' | '`')
    }) {
        if token.len() < 10 {
            continue;
        }
        if !token.contains("github.com")
            && !token.contains("gitlab.com")
            && !token.contains("bitbucket.org")
        {
            continue;
        }
        let cleaned = token.trim_end_matches(['.', ',', ';', ':', '!', '?', ')']);
        if let Some(repo) = parse_repo_url(cleaned) {
            let key = format!(
                "{}/{}",
                repo.owner.to_ascii_lowercase(),
                repo.repo.to_ascii_lowercase()
            );
            if seen.insert(key) {
                out.push(repo);
            }
        }
    }
    out
}

/// Validate a URL with the Layer A parser — single source of truth for
/// what counts as a repo URL.
fn parse_repo_url(url: &str) -> Option<DetectedRepo> {
    use edgequake_pdf::{detect_repos, LinkExtraction, PdfLinkAnnotation};

    let ext = LinkExtraction {
        links: vec![PdfLinkAnnotation {
            url: url.to_string(),
            page_index: 0,
            y_top: 0.0,
        }],
        refs_boundary: None,
        page_count: 1,
    };
    detect_repos(&ext).into_iter().next()
}

/// Build the LLM tie-breaker prompt. Each shortlisted hit contributes a
/// block tagged with its source (`github_api` / `web_search`) so the LLM
/// can prefer authoritative origins on ties.
fn build_llm_prompt(
    title: &str,
    author: Option<&str>,
    shortlist: &[(Hit, String)],
    config: &RepoResolverConfig,
) -> String {
    let mut prompt = String::with_capacity(config.llm_prompt_char_budget + 512);
    prompt.push_str(
        "You are given a research paper's title, author, and the most relevant code repositories surfaced by both the GitHub Search API and a SearXNG meta-search. \
         Pick the single best repo that implements THIS paper's method. \
         \n\n\
         Eligible:\n\
         - Code released by the paper's authors (preferred when present — check repo owner vs. author surname)\n\
         - A faithful third-party reimplementation whose README explicitly cites this paper by title, arXiv id, or author names\n\
         \n\
         NOT eligible:\n\
         - Widely-used libraries the paper happens to use or import (langchain, pytorch, numpy, transformers, scikit-learn, tensorflow, jax, …)\n\
         - Repos of baseline / prior-art methods the paper compares against\n\
         - Dataset or benchmark repos\n\
         - `awesome-*` curated lists\n\n",
    );
    prompt.push_str("Paper title: ");
    prompt.push_str(title);
    prompt.push('\n');
    if let Some(a) = author {
        prompt.push_str("First author: ");
        prompt.push_str(a);
        prompt.push('\n');
    }
    prompt.push_str("\n--- CANDIDATES ---\n\n");

    let per_hit = config.chars_per_hit_for_llm;
    let mut remaining = config.llm_prompt_char_budget.saturating_sub(prompt.len());
    for (hit, readme) in shortlist {
        if remaining < 200 {
            break;
        }
        let cap = per_hit.min(remaining.saturating_sub(200));
        let clipped: String = readme.chars().take(cap).collect();
        let desc = hit.inner.description.as_deref().unwrap_or("");
        let source_tag = match hit.source {
            HitSource::GithubApi => "github_api",
            HitSource::WebSearch => "web_search",
        };
        let chunk = format!(
            "[#{rank} {source} {url} — {desc} (★{stars})]\n{clipped}\n\n",
            rank = hit.inner.rank,
            source = source_tag,
            url = hit.inner.html_url,
            desc = desc,
            stars = hit.inner.stargazers_count,
        );
        remaining = remaining.saturating_sub(chunk.len());
        prompt.push_str(&chunk);
    }

    prompt.push_str(
        "\n--- INSTRUCTIONS ---\n\
         Respond with ONLY the URL in the form https://github.com/OWNER/REPO. \
         If none of the listed repos clearly implements the paper, respond with NONE.\n",
    );
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_hit(rank: usize, owner: &str, repo: &str, source: HitSource) -> Hit {
        Hit {
            inner: GithubSearchHit {
                owner: owner.into(),
                repo: repo.into(),
                html_url: format!("https://github.com/{owner}/{repo}"),
                description: Some("a test repo".into()),
                stargazers_count: 0,
                rank,
            },
            source,
        }
    }

    #[test]
    fn merge_dedupes_across_sources() {
        // Same repo seen in both layers: keep only one entry, attribute
        // to whichever came first (GitHub iterates first in merge_hits).
        let gh = vec![mk_hit(0, "wuwuz", "Pacmann", HitSource::GithubApi)];
        let web = vec![
            mk_hit(0, "wuwuz", "Pacmann", HitSource::WebSearch),
            mk_hit(1, "other", "thing", HitSource::WebSearch),
        ];
        let merged = merge_hits(gh, web, 10);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].source, HitSource::GithubApi);
        assert_eq!(merged[0].inner.owner, "wuwuz");
        assert_eq!(merged[1].inner.owner, "other");
    }

    #[test]
    fn merge_dedup_is_case_insensitive() {
        let gh = vec![mk_hit(0, "WUWUZ", "PACMANN", HitSource::GithubApi)];
        let web = vec![mk_hit(0, "wuwuz", "Pacmann", HitSource::WebSearch)];
        let merged = merge_hits(gh, web, 10);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn merge_caps_at_max() {
        let gh: Vec<Hit> = (0..20)
            .map(|i| mk_hit(i, &format!("o{i}"), &format!("r{i}"), HitSource::GithubApi))
            .collect();
        let merged = merge_hits(gh, vec![], 5);
        assert_eq!(merged.len(), 5);
    }

    #[test]
    fn site_github_query_drops_stopwords() {
        let q = build_site_github_query(
            "Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform",
        );
        assert!(q.starts_with("site:github.com "));
        assert!(q.contains("Stained"));
        assert!(q.contains("Glass"));
        assert!(q.contains("Transform"));
        assert!(!q.contains(" Of "));
    }

    #[test]
    fn scan_finds_github_url_in_prose() {
        let md = "See the implementation at https://github.com/foo/bar for details.";
        let r = scan_markdown_for_repos(md);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].owner, "foo");
        assert_eq!(r[0].repo, "bar");
    }

    #[test]
    fn scan_dedupes_repeated_url() {
        let md = "[A](https://github.com/foo/bar) and again at https://github.com/foo/bar/tree/main";
        let r = scan_markdown_for_repos(md);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn llm_prompt_tags_source_per_hit() {
        let cfg = RepoResolverConfig::default();
        let shortlist = vec![
            (
                mk_hit(0, "wuwuz", "Pacmann", HitSource::GithubApi),
                "official paper repo".into(),
            ),
            (
                mk_hit(1, "other", "thing", HitSource::WebSearch),
                "found via arxiv".into(),
            ),
        ];
        let p = build_llm_prompt("PACMANN paper", Some("Mingxun Zhou"), &shortlist, &cfg);
        assert!(p.contains("github_api"));
        assert!(p.contains("web_search"));
        assert!(p.contains("wuwuz/Pacmann"));
        assert!(p.contains("respond with NONE"));
    }

    #[test]
    fn llm_prompt_respects_char_budget() {
        let cfg = RepoResolverConfig {
            max_github_hits: 5,
            max_web_search_pages: 4,
            chars_per_hit_for_llm: 5_000,
            llm_prompt_char_budget: 2_000,
            max_shortlist_size: 8,
        };
        let big = "x".repeat(50_000);
        let shortlist = vec![
            (mk_hit(0, "alice", "alpha", HitSource::GithubApi), big.clone()),
            (mk_hit(1, "bob", "beta", HitSource::WebSearch), big),
        ];
        let p = build_llm_prompt("Title", Some("A"), &shortlist, &cfg);
        assert!(p.len() <= cfg.llm_prompt_char_budget + 500);
    }
}
