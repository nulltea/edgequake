//! Orchestrator: paper markdown → candidate reference repository URL.
//!
//! Flow:
//! 1. [`extract_front_matter`] on the markdown → title + first author.
//! 2. SearXNG query `"{first_author} {title} github"` (or just `"{title} github"`
//!    when author extraction fails).
//! 3. Sequential Crawl4AI fetch of top-N result pages, with early exit: if
//!    any page's raw markdown already contains a valid repo URL we skip the LLM.
//! 4. LLM tie-breaker: if step 3 surfaces zero or multiple unique repos, ask
//!    the workspace LLM to pick the canonical one.
//! 5. Validate the chosen URL with [`edgequake_pdf::repos`] (the same parser
//!    used by Layer A — one source of truth for what counts as a "repo URL").

use std::sync::Arc;

use edgequake_llm::traits::LLMProvider;
use edgequake_pdf::{DetectedRepo, RepoHost};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

use super::front_matter::extract_front_matter;
use super::{Crawl4aiClient, Crawl4aiError, SearxngClient, SearxngError};

#[derive(Debug, Error)]
pub enum RepoResolverError {
    #[error("front-matter extraction failed: could not find a title in the paper markdown")]
    NoFrontMatter,
    #[error(transparent)]
    Searxng(#[from] SearxngError),
    #[error(transparent)]
    Crawl4ai(#[from] Crawl4aiError),
    #[error("LLM call failed: {0}")]
    Llm(String),
    #[error("resolver found no repository candidates")]
    NotFound,
}

#[derive(Debug, Clone)]
pub struct RepoResolverConfig {
    /// Max SearXNG results to fetch via Crawl4AI.
    pub max_pages_to_crawl: usize,
    /// Hard cap on markdown characters sent to the LLM per page.
    pub chars_per_page_for_llm: usize,
    /// Upper bound on the final prompt string (concat of up to `max_pages`).
    pub llm_prompt_char_budget: usize,
}

impl Default for RepoResolverConfig {
    fn default() -> Self {
        Self {
            // Bumped 3 → 6 to catch third-party reimplementations that sit
            // below the arxiv/huggingface/semanticscholar pages which
            // usually dominate the top 3 SearXNG hits. The extra 3 crawls
            // double Layer B latency but we were missing correct repos
            // that were just outside the previous cap.
            max_pages_to_crawl: 6,
            chars_per_page_for_llm: 6_000,
            llm_prompt_char_budget: 16_000,
        }
    }
}

/// The output surfaced from the resolver. Intentionally identical in shape to
/// [`edgequake_pdf::DetectedRepo`] except:
/// - `page_index`/`y_top` don't apply (web source), so we store the SearXNG
///   result rank the URL was found in.
/// - A `source_url` lets the UI show *how* we got here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedRepo {
    pub host: RepoHost,
    pub owner: String,
    pub repo: String,
    pub url: String,
    /// 0-based rank of the SearXNG result page this URL surfaced from (if
    /// known). `None` means the LLM resolved it without a specific source
    /// (e.g. inferred from abstract-level context).
    pub source_rank: Option<usize>,
    /// The SearXNG result URL we fetched. For auditability in the UI.
    pub source_url: Option<String>,
}

impl From<DetectedRepo> for ResolvedRepo {
    fn from(r: DetectedRepo) -> Self {
        Self {
            host: r.host,
            owner: r.owner,
            repo: r.repo,
            url: r.url,
            source_rank: None,
            source_url: None,
        }
    }
}

/// Run the three-step web-search pipeline on a paper.
///
/// Returns `Err(NoFrontMatter)` if we can't find a title (caller can decide
/// whether to retry with LLM-assisted extraction), or `Err(NotFound)` if no
/// plausible repo was surfaced.
pub async fn resolve_repo(
    paper_markdown: &str,
    searxng: &SearxngClient,
    crawl: &Crawl4aiClient,
    llm: Arc<dyn LLMProvider>,
    config: &RepoResolverConfig,
) -> Result<ResolvedRepo, RepoResolverError> {
    let fm = extract_front_matter(paper_markdown).ok_or(RepoResolverError::NoFrontMatter)?;

    // Run two SearXNG queries in parallel and merge results:
    //
    // - PRIMARY  : "<author> <title> github" — same as before; good for
    //              papers whose authors actually released code (title hit
    //              on arxiv's "Code" sidebar, author's GitHub profile, etc.)
    // - FALLBACK : "site:github.com <title-core>" — forces all results to
    //              be actual github.com pages; crucial for third-party
    //              reimplementations that don't surface in top-N on the
    //              primary query.
    //
    // We merge by URL and keep the best (lowest) rank across the two. The
    // fast-path / crawl / LLM tie-breaker below see a single combined list,
    // so no other logic has to change.
    let primary_q = build_query(&fm.title, fm.first_author.as_deref());
    let fallback_q = build_site_github_query(&fm.title);
    debug!(primary = %primary_q, fallback = %fallback_q, "searxng queries");

    let (primary, fallback) = tokio::join!(
        searxng.search(&primary_q),
        searxng.search(&fallback_q)
    );
    let primary = primary?;
    // Fallback is best-effort: a failure on it doesn't fail the whole
    // resolver — the primary query usually carries us.
    let fallback = fallback.unwrap_or_default();

    let results = merge_searxng_results(primary, fallback);
    if results.is_empty() {
        return Err(RepoResolverError::NotFound);
    }

    // Fast path: if SearXNG itself returned a github.com URL that parses to a
    // valid owner/repo, skip crawling entirely.
    for (rank, r) in results.iter().enumerate() {
        if let Some(repo) = parse_repo_url(&r.url) {
            return Ok(ResolvedRepo {
                source_rank: Some(rank),
                source_url: Some(r.url.clone()),
                ..repo.into()
            });
        }
    }

    // Crawl top-N pages; if the cleaned markdown of any page contains a repo
    // URL we can validate ourselves, that's our answer — no LLM needed.
    let mut crawled: Vec<(usize, String, String)> = Vec::new(); // (rank, source_url, markdown)
    for (rank, r) in results.iter().enumerate().take(config.max_pages_to_crawl) {
        match crawl.markdown(&r.url).await {
            Ok(resp) if resp.success => {
                if let Some(repo) = scan_markdown_for_repo(&resp.markdown) {
                    return Ok(ResolvedRepo {
                        source_rank: Some(rank),
                        source_url: Some(r.url.clone()),
                        ..repo.into()
                    });
                }
                crawled.push((rank, r.url.clone(), resp.markdown));
            }
            Ok(resp) => {
                warn!(url = %r.url, "crawl4ai returned success=false");
                crawled.push((rank, r.url.clone(), resp.markdown));
            }
            Err(e) => {
                warn!(url = %r.url, error = %e, "crawl4ai failed");
            }
        }
    }

    if crawled.is_empty() {
        return Err(RepoResolverError::NotFound);
    }

    // LLM tie-breaker.
    let prompt = build_llm_prompt(&fm.title, fm.first_author.as_deref(), &crawled, config);
    let resp = llm
        .complete(&prompt)
        .await
        .map_err(|e| RepoResolverError::Llm(e.to_string()))?;
    let raw = llm_output_text(&resp);

    match parse_repo_url(raw.trim()) {
        Some(repo) => {
            // Find the source we gave the LLM that matches the chosen URL —
            // purely for UI attribution; not required for correctness.
            let (source_rank, source_url) = crawled
                .iter()
                .find(|(_, _, md)| md.contains(&repo.url))
                .map(|(r, u, _)| (Some(*r), Some(u.clone())))
                .unwrap_or((None, None));
            Ok(ResolvedRepo {
                source_rank,
                source_url,
                ..repo.into()
            })
        }
        None => Err(RepoResolverError::NotFound),
    }
}

fn build_query(title: &str, author: Option<&str>) -> String {
    match author {
        Some(a) => format!("{a} {title} github"),
        None => format!("{title} github"),
    }
}

/// Build a `site:github.com`-scoped SearXNG query using just the most
/// content-bearing words of the title. Many third-party reimplementation
/// repos don't mention the full title verbatim, so we clip to ~8 tokens
/// after dropping stopwords. `site:github.com` forces every hit to be a
/// github page; combined with the fast-path (returns on first parseable
/// github.com/{owner}/{repo}) this typically surfaces the repo at rank 0.
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
        // Degenerate input — fall back to raw title with the operator.
        return format!("site:github.com {title}");
    }
    format!("site:github.com {}", core.join(" "))
}

/// Deduplicate SearXNG results across two queries, preserving the best
/// (lowest) rank of each URL. Primary-query rank wins ties because it
/// carries the author signal.
fn merge_searxng_results(
    primary: Vec<super::searxng::SearxngResult>,
    fallback: Vec<super::searxng::SearxngResult>,
) -> Vec<super::searxng::SearxngResult> {
    use std::collections::HashMap;
    let mut best_rank: HashMap<String, (usize, super::searxng::SearxngResult)> = HashMap::new();
    for (rank, r) in primary.into_iter().enumerate() {
        best_rank.entry(r.url.clone()).or_insert((rank, r));
    }
    for (rank, r) in fallback.into_iter().enumerate() {
        best_rank
            .entry(r.url.clone())
            .and_modify(|(existing_rank, _)| {
                if rank < *existing_rank {
                    *existing_rank = rank;
                }
            })
            .or_insert((rank, r));
    }
    let mut merged: Vec<(usize, super::searxng::SearxngResult)> =
        best_rank.into_values().collect();
    merged.sort_by_key(|(r, _)| *r);
    merged.into_iter().map(|(_, r)| r).collect()
}

/// Scan free-form markdown for an embedded repo URL.
///
/// We simply re-use the Layer A parser: tokenise on whitespace + punctuation,
/// try each token as a URL, keep the first one that parses.
fn scan_markdown_for_repo(md: &str) -> Option<DetectedRepo> {
    // Split on whitespace and a few markdown delimiters.
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
        // Strip trailing punctuation that commonly attaches to URLs in prose.
        let cleaned = token.trim_end_matches(['.', ',', ';', ':', '!', '?', ')']);
        if let Some(repo) = parse_repo_url(cleaned) {
            return Some(repo);
        }
    }
    None
}

/// Validate a URL with the Layer A parser. Goes through a minimal
/// [`edgequake_pdf::LinkExtraction`] synthetic input to share one source of
/// truth with Layer A for what counts as a repo URL.
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

fn llm_output_text(resp: &edgequake_llm::LLMResponse) -> String {
    // LLMResponse in this crate version exposes `content: String`.
    resp.content.clone()
}

fn build_llm_prompt(
    title: &str,
    author: Option<&str>,
    pages: &[(usize, String, String)],
    config: &RepoResolverConfig,
) -> String {
    let mut prompt = String::with_capacity(config.llm_prompt_char_budget + 512);
    prompt.push_str(
        "You are given a research paper's title, author, and excerpts from web pages found via search. \
        Identify the single best GitHub, GitLab, or Bitbucket repository URL that implements THIS paper's method. \
        \n\n\
        Eligible:\n\
        - Code released by the paper's authors (preferred when present)\n\
        - A faithful third-party reimplementation whose README explicitly cites this paper by title, arXiv id, or author names\n\
        \n\
        NOT eligible (these are tooling / dependencies, not implementations):\n\
        - Widely-used libraries the paper happens to use or import (langchain, pytorch, numpy, transformers, scikit-learn, tensorflow, jax, …)\n\
        - Repos of baseline / prior-art methods the paper compares against\n\
        - Dataset or benchmark repos\n\
        - An author's generic GitHub profile page (e.g. github.com/username with no repo path)\n\n",
    );
    prompt.push_str("Paper title: ");
    prompt.push_str(title);
    prompt.push('\n');
    if let Some(a) = author {
        prompt.push_str("First author: ");
        prompt.push_str(a);
        prompt.push('\n');
    }
    prompt.push_str("\n--- WEB EXCERPTS ---\n\n");

    let per_page = config.chars_per_page_for_llm;
    let mut remaining = config.llm_prompt_char_budget.saturating_sub(prompt.len());
    for (rank, url, md) in pages {
        if remaining < 200 {
            break;
        }
        let cap = per_page.min(remaining.saturating_sub(120));
        let clipped: String = md.chars().take(cap).collect();
        let chunk = format!("[Source #{rank}: {url}]\n{clipped}\n\n");
        remaining = remaining.saturating_sub(chunk.len());
        prompt.push_str(&chunk);
    }

    prompt.push_str(
        "\n--- INSTRUCTIONS ---\n\
        Respond with ONLY the URL in the form https://github.com/OWNER/REPO \
        (or gitlab.com / bitbucket.org equivalent). If no eligible repository is present, respond with NONE.\n",
    );
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_with_author() {
        assert_eq!(build_query("Paper", Some("Alice")), "Alice Paper github");
    }

    #[test]
    fn query_without_author() {
        assert_eq!(build_query("Paper", None), "Paper github");
    }

    #[test]
    fn scans_markdown_for_github_url() {
        let md = "See the implementation at https://github.com/foo/bar for details.";
        let r = scan_markdown_for_repo(md).unwrap();
        assert_eq!(r.owner, "foo");
        assert_eq!(r.repo, "bar");
    }

    #[test]
    fn scans_ignores_refs_to_other_hosts() {
        let md = "Available at https://example.com/foo/bar and nothing else.";
        assert!(scan_markdown_for_repo(md).is_none());
    }

    #[test]
    fn scans_handles_markdown_link_syntax() {
        let md = "Check the [repo](https://github.com/foo/bar) for code.";
        let r = scan_markdown_for_repo(md).unwrap();
        assert_eq!(r.owner, "foo");
        assert_eq!(r.repo, "bar");
    }

    #[test]
    fn scans_strips_trailing_punct() {
        let md = "The code is at https://github.com/foo/bar. Thanks!";
        let r = scan_markdown_for_repo(md).unwrap();
        assert_eq!(r.repo, "bar");
    }

    #[test]
    fn site_github_query_drops_stopwords_and_clips() {
        let q = build_site_github_query(
            "Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform",
        );
        // Starts with the operator
        assert!(q.starts_with("site:github.com "));
        // Common stopwords ("of") are dropped
        assert!(!q.contains(" Of ") && !q.contains(" of "));
        // Includes the distinctive words
        assert!(q.contains("Stained"));
        assert!(q.contains("Glass"));
        assert!(q.contains("Transform"));
    }

    #[test]
    fn site_github_query_degenerate_title() {
        let q = build_site_github_query("a, b, c;");
        // Nothing substantive — falls back to the raw title after the operator
        assert!(q.starts_with("site:github.com "));
    }

    #[test]
    fn merge_preserves_best_rank_across_queries() {
        use super::super::searxng::SearxngResult;
        let mk = |u: &str| SearxngResult {
            url: u.into(),
            title: u.into(),
            content: String::new(),
        };
        // Primary has foo/bar buried at rank 3; fallback surfaces it at
        // rank 0. Assertions target the merge contract (dedup + best
        // rank wins), not the exact tie-break order between same-rank
        // URLs — HashMap iteration isn't deterministic.
        let p = vec![
            mk("https://arxiv.org/abs/1"),
            mk("https://b"),
            mk("https://c"),
            mk("https://github.com/foo/bar"),
        ];
        let f = vec![
            mk("https://github.com/foo/bar"),
            mk("https://github.com/other/thing"),
        ];
        let merged = merge_searxng_results(p, f);

        // foo/bar moved up to rank 0 → it's among the top-1 group
        // (a rank-0 URL is always in position 0 or 1 of the sorted output
        // because the only other rank-0 URL is arxiv from primary).
        let foo_bar_pos = merged
            .iter()
            .position(|r| r.url == "https://github.com/foo/bar")
            .expect("foo/bar should be present");
        assert!(foo_bar_pos <= 1, "foo/bar at position {foo_bar_pos}");

        // All primary URLs preserved.
        assert!(merged.iter().any(|r| r.url == "https://arxiv.org/abs/1"));
        assert!(merged
            .iter()
            .any(|r| r.url == "https://github.com/other/thing"));

        // No duplicate URLs.
        let unique: std::collections::HashSet<_> = merged.iter().map(|r| &r.url).collect();
        assert_eq!(unique.len(), merged.len());
    }

    #[test]
    fn llm_prompt_respects_char_budget() {
        let cfg = RepoResolverConfig {
            max_pages_to_crawl: 3,
            chars_per_page_for_llm: 5000,
            llm_prompt_char_budget: 2000,
        };
        let big = "x".repeat(50_000);
        let pages = vec![
            (0, "https://a".into(), big.clone()),
            (1, "https://b".into(), big),
        ];
        let prompt = build_llm_prompt("Title", Some("A"), &pages, &cfg);
        assert!(prompt.len() <= cfg.llm_prompt_char_budget + 500);
    }
}
