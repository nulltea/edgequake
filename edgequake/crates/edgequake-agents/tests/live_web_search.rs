//! Live smoke tests against the user's SearXNG + Crawl4AI deployments.
//!
//! Skipped unless `SEARXNG_URL` and `CRAWL4AI_URL` are set. No LLM call —
//! we check the "fast path" (direct SearXNG repo result) and the "markdown
//! scan" path (repo URL found inside a crawled page). Gives us confidence
//! that the HTTP contracts haven't drifted before we wire the task up.

use std::sync::Arc;

use edgequake_agents::web_search::{
    resolve_repo, Crawl4aiClient, RepoResolverConfig, SearxngClient,
};
use edgequake_llm::MockProvider;

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

#[tokio::test]
async fn searxng_returns_results_for_known_query() {
    let Some(url) = env("SEARXNG_URL") else {
        eprintln!("SKIP: SEARXNG_URL not set");
        return;
    };
    let client = SearxngClient::new(url);
    let results = client
        .search("swiftide bosun-ai rust github")
        .await
        .expect("searxng search");
    println!("got {} results", results.len());
    for (i, r) in results.iter().take(5).enumerate() {
        println!("  #{i}: {}", r.url);
    }
    assert!(!results.is_empty(), "expected at least one result");
    // At least one result should reference swiftide on github.
    let has_swiftide = results.iter().any(|r| {
        r.url.contains("github.com/bosun-ai/swiftide")
            || r.title.to_lowercase().contains("swiftide")
    });
    assert!(has_swiftide, "expected swiftide repo somewhere in results");
}

#[tokio::test]
async fn crawl4ai_returns_markdown_for_github_page() {
    let Some(url) = env("CRAWL4AI_URL") else {
        eprintln!("SKIP: CRAWL4AI_URL not set");
        return;
    };
    let client = Crawl4aiClient::new(url);
    let resp = client
        .markdown("https://github.com/bosun-ai/swiftide")
        .await
        .expect("crawl markdown");
    println!(
        "success={} url={} md_len={}",
        resp.success,
        resp.url,
        resp.markdown.len()
    );
    assert!(resp.markdown.contains("swiftide"));
}

/// Full Layer B: given a real paper's markdown, find its reference repo.
///
/// Exercises the fast path (SearXNG returns a github URL at rank 0) and/or the
/// markdown-scan path. The LLM is mocked — if we reach that path the test
/// fails, which is deliberate: for a well-known paper we expect the non-LLM
/// paths to hit.
#[tokio::test]
async fn resolver_finds_repo_for_alphaevolve_paper() {
    let (Some(sx_url), Some(cr_url)) = (env("SEARXNG_URL"), env("CRAWL4AI_URL")) else {
        eprintln!("SKIP: SEARXNG_URL / CRAWL4AI_URL not set");
        return;
    };
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let md_path = std::path::PathBuf::from(crate_dir)
        .join("../../../legacy/edgequake-pdf/test-data/real_dataset/AlphaEvolve.md");
    if !md_path.exists() {
        eprintln!("SKIP: AlphaEvolve.md fixture not found at {md_path:?}");
        return;
    }
    let md = std::fs::read_to_string(&md_path).expect("read md");

    let sx = SearxngClient::new(sx_url);
    let cr = Crawl4aiClient::new(cr_url);
    let llm: Arc<dyn edgequake_llm::traits::LLMProvider> = Arc::new(MockProvider::new());
    let cfg = RepoResolverConfig::default();

    let resolved = resolve_repo(&md, &sx, &cr, llm, &cfg)
        .await
        .expect("resolver should find a repo");

    println!(
        "resolved: {:?} {}/{} -> {} (source_rank={:?} source_url={:?})",
        resolved.host,
        resolved.owner,
        resolved.repo,
        resolved.url,
        resolved.source_rank,
        resolved.source_url
    );

    // AlphaEvolve's reference implementations live under the google-deepmind
    // org; accept the known repos plus any related GDM repo that surfaces top.
    assert!(
        matches!(resolved.host, edgequake_pdf::RepoHost::GitHub),
        "expected a github repo, got {:?}",
        resolved.host
    );
    let owner_lower = resolved.owner.to_ascii_lowercase();
    assert!(
        owner_lower == "google-deepmind" || owner_lower == "deepmind" || owner_lower == "google",
        "expected a DeepMind/Google org, got owner={:?}",
        resolved.owner
    );
}

/// Pure search-quality check: given a paper's title + first author, does
/// SearXNG find the paper (or its project/repo page) online? Exercises Layer
/// B's `web.search` step in isolation — no front-matter extraction, no
/// resolver downstream — so failures pin blame on the search layer alone.
#[tokio::test]
async fn searxng_finds_known_papers() {
    let Some(url) = env("SEARXNG_URL") else {
        eprintln!("SKIP: SEARXNG_URL not set");
        return;
    };
    let client = SearxngClient::new(url);

    // Fixtures: (title, first_author, expected-substring-in-any-result-URL)
    let cases = [
        (
            "SpaceTimePilot: Generative Rendering of Dynamic Scenes",
            "Zhening Huang",
            &["zheninghuang", "space-time-pilot", "2512.25075"][..],
        ),
        (
            "AlphaEvolve: A coding agent for scientific and algorithmic discovery",
            "Alexander Novikov",
            &["alphaevolve", "google-deepmind", "deepmind.google"][..],
        ),
    ];

    for (title, author, expected) in cases {
        let q = format!("{author} {title} github");
        println!("\n--- query: {q:?} ---");
        let results = client.search(&q).await.expect("searxng search");
        println!("got {} results", results.len());
        for (i, r) in results.iter().take(10).enumerate() {
            println!("  #{i}: {}", r.url);
        }
        let hit = results.iter().any(|r| {
            let u = r.url.to_lowercase();
            expected.iter().any(|e| u.contains(e))
        });
        assert!(
            hit,
            "expected at least one result URL to contain one of {expected:?} for query {q:?}"
        );
    }
}

/// A paper with a project-page URL but no GitHub link. Layer A returns nothing;
/// Layer B might still find the repo via search — or correctly return NotFound.
/// We accept either, but print the outcome so regressions are visible.
#[tokio::test]
async fn resolver_behaviour_for_project_page_paper() {
    let (Some(sx_url), Some(cr_url)) = (env("SEARXNG_URL"), env("CRAWL4AI_URL")) else {
        eprintln!("SKIP: SEARXNG_URL / CRAWL4AI_URL not set");
        return;
    };
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let md_path = std::path::PathBuf::from(crate_dir)
        .join("../../../legacy/edgequake-pdf/test-data/real_dataset/01_2512.25075v1.md");
    if !md_path.exists() {
        eprintln!("SKIP: fixture not found");
        return;
    }
    let md = std::fs::read_to_string(&md_path).expect("read md");

    let sx = SearxngClient::new(sx_url);
    let cr = Crawl4aiClient::new(cr_url);
    let llm: Arc<dyn edgequake_llm::traits::LLMProvider> = Arc::new(MockProvider::new());
    let cfg = RepoResolverConfig::default();

    let resolved = resolve_repo(&md, &sx, &cr, llm, &cfg)
        .await
        .expect("resolver should find the SpaceTimePilot repo via web search");
    println!(
        "project-page paper resolved to: {}/{} (rank={:?})",
        resolved.owner, resolved.repo, resolved.source_rank
    );
    let owner_lower = resolved.owner.to_ascii_lowercase();
    let repo_lower = resolved.repo.to_ascii_lowercase();
    assert!(
        owner_lower.contains("zheninghuang")
            || repo_lower.contains("spacetimepilot")
            || repo_lower.contains("space-time-pilot"),
        "expected SpaceTimePilot repo by author zheninghuang, got {}/{}",
        resolved.owner,
        resolved.repo
    );
}
