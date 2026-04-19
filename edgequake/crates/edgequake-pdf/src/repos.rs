//! Reference-repository detection: filter raw PDF link annotations down to
//! a ranked list of candidate `{org}/{repo}` URLs that plausibly represent
//! the paper's own reference implementation.
//!
//! Applied filters, in order:
//! 1. Host allowlist (`github.com`, `gitlab.com`, `bitbucket.org`).
//! 2. Reference-section exclusion — links past the bibliography cite *other*
//!    papers' repos, not the author's own.
//! 3. Shape filter — URL must have at least `/{owner}/{repo}` path segments.
//! 4. Dedup — one candidate per `(host, owner, repo)`.
//! 5. Ranking — links on earlier pages outrank links on later pages (abstract
//!    and intro are the usual self-citation locations).

use serde::{Deserialize, Serialize};

use crate::links::LinkExtraction;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RepoHost {
    GitHub,
    GitLab,
    Bitbucket,
}

impl RepoHost {
    pub fn as_str(self) -> &'static str {
        match self {
            RepoHost::GitHub => "github.com",
            RepoHost::GitLab => "gitlab.com",
            RepoHost::Bitbucket => "bitbucket.org",
        }
    }
}

/// Classify a URL and return `(host, path_after_host)`.
///
/// Recognises plain repo hosts plus two wrapper patterns commonly seen in
/// papers:
/// - `raw.githubusercontent.com/{owner}/{repo}/...`
/// - `colab.research.google.com/github/{owner}/{repo}/...`
fn classify_url(url: &str) -> Option<(RepoHost, String)> {
    let lower = url.to_ascii_lowercase();

    // Direct hosts — find as a marker and split after it.
    for (marker, host) in [
        ("github.com/", RepoHost::GitHub),
        ("gitlab.com/", RepoHost::GitLab),
        ("bitbucket.org/", RepoHost::Bitbucket),
    ] {
        if let Some(idx) = lower.find(marker) {
            // Guard against false matches like "mygithub.com" — the marker must
            // be at the start of a host component.
            let before_ok = idx == 0 || {
                let prev = lower.as_bytes()[idx - 1];
                prev == b'/' || prev == b'.' || prev == b'@'
            };
            if before_ok {
                let path_start = idx + marker.len();
                return Some((host, lower[path_start..].to_string()));
            }
        }
    }

    // Colab wrapper: colab.research.google.com/github/{owner}/{repo}/...
    if let Some(idx) = lower.find("colab.research.google.com/github/") {
        let path_start = idx + "colab.research.google.com/github/".len();
        return Some((RepoHost::GitHub, lower[path_start..].to_string()));
    }

    // Raw GitHub content: raw.githubusercontent.com/{owner}/{repo}/{branch}/...
    if let Some(idx) = lower.find("raw.githubusercontent.com/") {
        let path_start = idx + "raw.githubusercontent.com/".len();
        return Some((RepoHost::GitHub, lower[path_start..].to_string()));
    }

    None
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedRepo {
    pub host: RepoHost,
    pub owner: String,
    pub repo: String,
    /// Canonical `https://{host}/{owner}/{repo}` URL.
    pub url: String,
    /// Page on which the link was found (zero-based, earliest occurrence).
    pub page_index: usize,
    /// y-top in PDF points on that page (larger = higher on the page).
    pub y_top: f32,
}

/// Detect candidate reference repositories in a link extraction.
///
/// Returned in ranked order (best first). Pass through an external scorer
/// (e.g. section-based re-ranking) to refine further if needed.
pub fn detect_repos(extraction: &LinkExtraction) -> Vec<DetectedRepo> {
    let mut seen: std::collections::HashMap<(RepoHost, String, String), DetectedRepo> =
        std::collections::HashMap::new();

    for link in &extraction.links {
        if extraction.is_past_refs(link) {
            continue;
        }
        let Some((host, rest)) = classify_url(&link.url) else {
            continue;
        };
        let Some((owner, repo)) = parse_owner_repo(&rest, host) else {
            continue;
        };
        let key = (host, owner.clone(), repo.clone());
        let candidate = DetectedRepo {
            host,
            owner: owner.clone(),
            repo: repo.clone(),
            url: format!("https://{}/{}/{}", host.as_str(), owner, repo),
            page_index: link.page_index,
            y_top: link.y_top,
        };
        seen.entry(key)
            .and_modify(|existing| {
                // Prefer earliest page; on same page, prefer higher y (earlier in reading order).
                let cur = (existing.page_index, -existing.y_top);
                let new = (candidate.page_index, -candidate.y_top);
                if new < cur {
                    *existing = candidate.clone();
                }
            })
            .or_insert(candidate);
    }

    let mut out: Vec<DetectedRepo> = seen.into_values().collect();
    // Earlier page wins; tie-break: higher y_top wins.
    out.sort_by(|a, b| {
        a.page_index
            .cmp(&b.page_index)
            .then_with(|| b.y_top.total_cmp(&a.y_top))
    });
    out
}

/// Parse `owner` and `repo` from a path already stripped of scheme+host.
///
/// Handles trailing `.git`, deep paths like `/owner/repo/tree/main/...`, and
/// query/fragment.
fn parse_owner_repo(path: &str, host: RepoHost) -> Option<(String, String)> {
    let path = path.trim_start_matches('/');
    // Strip fragment and query.
    let path = path.split('#').next().unwrap_or(path);
    let path = path.split('?').next().unwrap_or(path);

    let mut parts = path.split('/').filter(|s| !s.is_empty());
    let owner = parts.next()?;
    let repo_raw = parts.next()?;
    let repo = repo_raw.trim_end_matches(".git");

    if !is_valid_slug(owner) || !is_valid_slug(repo) {
        return None;
    }
    if is_reserved_owner_path(host, owner) {
        return None;
    }
    Some((owner.to_string(), repo.to_string()))
}

fn is_valid_slug(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 100
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
}

fn is_reserved_owner_path(host: RepoHost, owner: &str) -> bool {
    let lower = owner.to_ascii_lowercase();
    match host {
        RepoHost::GitHub => matches!(
            lower.as_str(),
            "search"
                | "about"
                | "pricing"
                | "features"
                | "enterprise"
                | "marketplace"
                | "topics"
                | "collections"
                | "trending"
                | "login"
                | "join"
                | "settings"
                | "notifications"
                | "issues"
                | "pulls"
                | "explore"
                | "sponsors"
                | "orgs"
                | "apps"
                | "contact"
                | "site"
                | "security"
                | "readme"
        ),
        RepoHost::GitLab => matches!(
            lower.as_str(),
            "help"
                | "explore"
                | "users"
                | "groups"
                | "-"
                | "admin"
                | "dashboard"
                | "profile"
                | "search"
        ),
        RepoHost::Bitbucket => matches!(
            lower.as_str(),
            "product" | "pricing" | "features" | "support" | "blog" | "account" | "repo"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::links::{LinkExtraction, PdfLinkAnnotation, ReferenceBoundary};

    fn link(url: &str, page: usize, y: f32) -> PdfLinkAnnotation {
        PdfLinkAnnotation {
            url: url.to_string(),
            page_index: page,
            y_top: y,
        }
    }

    fn classify_and_parse(url: &str) -> Option<(RepoHost, String, String)> {
        let (host, rest) = classify_url(url)?;
        let (owner, repo) = parse_owner_repo(&rest, host)?;
        Some((host, owner, repo))
    }

    #[test]
    fn parses_github_plain() {
        assert_eq!(
            classify_and_parse("https://github.com/foo/bar"),
            Some((RepoHost::GitHub, "foo".to_string(), "bar".to_string()))
        );
    }

    #[test]
    fn parses_github_with_deep_path() {
        assert_eq!(
            classify_and_parse("https://github.com/foo/bar/tree/main/src"),
            Some((RepoHost::GitHub, "foo".to_string(), "bar".to_string()))
        );
    }

    #[test]
    fn strips_git_suffix_and_query() {
        assert_eq!(
            classify_and_parse("https://github.com/foo/bar.git?ref=v1"),
            Some((RepoHost::GitHub, "foo".to_string(), "bar".to_string()))
        );
    }

    #[test]
    fn rejects_reserved_paths() {
        assert_eq!(classify_and_parse("https://github.com/search?q=rag"), None);
        assert_eq!(classify_and_parse("https://github.com/about"), None);
    }

    #[test]
    fn rejects_root() {
        assert_eq!(classify_and_parse("https://github.com/"), None);
        assert_eq!(classify_and_parse("https://github.com/foo"), None);
    }

    #[test]
    fn parses_colab_github_wrapper() {
        assert_eq!(
            classify_and_parse(
                "https://colab.research.google.com/github/google-deepmind/alphaevolve_results/blob/master/mathematical_results.ipynb"
            ),
            Some((
                RepoHost::GitHub,
                "google-deepmind".to_string(),
                "alphaevolve_results".to_string()
            ))
        );
    }

    #[test]
    fn parses_raw_githubusercontent() {
        assert_eq!(
            classify_and_parse("https://raw.githubusercontent.com/foo/bar/main/README.md"),
            Some((RepoHost::GitHub, "foo".to_string(), "bar".to_string()))
        );
    }

    #[test]
    fn rejects_lookalike_host() {
        assert_eq!(classify_and_parse("https://mygithub.com/foo/bar"), None);
        assert_eq!(classify_and_parse("https://notgithub.com/foo/bar"), None);
    }

    #[test]
    fn detects_and_ranks_by_page() {
        let ext = LinkExtraction {
            links: vec![
                link("https://github.com/late/one", 8, 500.0),
                link("https://github.com/early/first", 0, 700.0),
                link("https://example.com/not/a/repo", 0, 600.0),
            ],
            refs_boundary: None,
            page_count: 10,
        };
        let got = detect_repos(&ext);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].owner, "early");
        assert_eq!(got[1].owner, "late");
    }

    #[test]
    fn filters_out_links_past_references() {
        let ext = LinkExtraction {
            links: vec![
                link("https://github.com/author/paper", 1, 600.0),
                link("https://github.com/cited/work", 9, 500.0),
            ],
            refs_boundary: Some(ReferenceBoundary {
                page_index: 8,
                y_top: 700.0,
            }),
            page_count: 10,
        };
        let got = detect_repos(&ext);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].owner, "author");
    }

    #[test]
    fn dedups_same_repo_across_pages_keeping_earliest() {
        let ext = LinkExtraction {
            links: vec![
                link("https://github.com/author/paper", 3, 200.0),
                link("https://github.com/author/paper/tree/main", 0, 600.0),
            ],
            refs_boundary: None,
            page_count: 5,
        };
        let got = detect_repos(&ext);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].page_index, 0);
    }

    #[test]
    fn detects_gitlab_and_bitbucket() {
        let ext = LinkExtraction {
            links: vec![
                link("https://gitlab.com/group/subgroup-project", 0, 400.0),
                link("https://bitbucket.org/team/repo", 1, 400.0),
            ],
            refs_boundary: None,
            page_count: 3,
        };
        let got = detect_repos(&ext);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].host, RepoHost::GitLab);
        assert_eq!(got[1].host, RepoHost::Bitbucket);
    }
}
