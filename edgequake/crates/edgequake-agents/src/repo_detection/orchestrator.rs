//! Top-level repo-detection flow: Layer A (PDF annotations) → Layer B
//! (web-search fallback) → ranked candidates ready for persistence.
//!
//! Returns the raw candidate list; the caller (API task processor) handles
//! DB writes, status transitions, and confidence-based filtering.

use std::sync::Arc;

use edgequake_llm::traits::LLMProvider;
use tracing::{debug, info, warn};

use super::types::{Confidence, DetectionMethod, RepoCandidate, RepoHost};
use super::verify::{verify_candidate, VerificationVerdict};
use crate::web_search::{
    extract_front_matter, resolve_repo, Crawl4aiClient, GithubClient, HitSource,
    PaperFrontMatter, RepoResolverConfig, RepoResolverError, SearxngClient,
};

/// External dependencies for Layer B.
///
/// `github` drives the search now (replaces the SearXNG path that kept
/// hitting CAPTCHAs). `crawl4ai` is still here because the verifier uses
/// it for non-GitHub README fetches; `searxng` is retained for callers
/// that want it elsewhere but no longer drives repo discovery.
pub struct WebSearchClients {
    pub github: GithubClient,
    pub searxng: SearxngClient,
    pub crawl4ai: Crawl4aiClient,
    pub llm: Arc<dyn LLMProvider>,
}

#[derive(Debug, Clone, Default)]
pub struct RepoDetectionConfig {
    /// Layer A: drop low-value annotation candidates (e.g. links found only
    /// after the references section). Carried through internally via
    /// [`edgequake_pdf::detect_repos`]; no knob today.
    _private: (),
}

#[derive(Debug, thiserror::Error)]
pub enum RepoDetectionError {
    #[error("Layer A PDF parsing failed: {0}")]
    PdfParse(String),
    #[error("Layer B web-search failed: {0}")]
    WebSearch(#[from] RepoResolverError),
}

/// Run Layer A against `pdf_bytes`, then verify, then maybe fall through to
/// Layer B using `paper_markdown` + `web_clients`.
///
/// Flow:
/// 1. Layer A: parse PDF for repo URLs before the references section.
/// 2. Verifier pass: LLM scores each candidate against paper metadata + repo
///    README. Confidence is re-derived from the verdict
///    (`official` → High, `third_party`/`inconclusive` → Medium,
///    `unrelated` → Low), so an unconditional `Confidence::High` from
///    Layer A can't outlive a "this README has nothing to do with the
///    paper" verdict.
/// 3. Layer B fallback: if Layer A produced candidates but none came out
///    `official`/`third_party` after verification, run web search once
///    and append its result (also verified). Layer B is **not** retried
///    when the no-Layer-A branch already ran it — a low-confidence
///    Layer B result is the best we can do, retrying would just loop.
///
/// The verifier never deletes candidates: low-confidence rows still
/// persist so reviewers can see *what* the PDF link pointed at, alongside
/// the web-search alternative. The workspace-level
/// `accept_unofficial_implementations` filter (applied by the caller)
/// decides whether `third_party`/`unrelated` rows are surfaced.
///
/// `pdf_bytes` may be empty (e.g. non-PDF ingestion); in that case we skip
/// Layer A and only run Layer B.
pub async fn run_detection(
    pdf_bytes: &[u8],
    paper_markdown: &str,
    web_clients: Option<&WebSearchClients>,
    _config: &RepoDetectionConfig,
) -> Result<DetectionOutcome, RepoDetectionError> {
    let layer_a = if pdf_bytes.is_empty() {
        Vec::new()
    } else {
        run_layer_a(pdf_bytes).await?
    };

    let layer_a_had_results = !layer_a.is_empty();
    let mut candidates: Vec<RepoCandidate>;
    let mut layer_b_attempted = false;

    if layer_a_had_results {
        info!(count = layer_a.len(), "layer_a found reference repos");
        candidates = layer_a;
    } else if let Some(clients) = web_clients {
        debug!("layer_a empty; falling through to layer_b");
        layer_b_attempted = true;
        match run_layer_b(paper_markdown, clients).await {
            Ok(Some(c)) => {
                info!("layer_b found a reference repo");
                candidates = vec![c];
            }
            Ok(None) => {
                info!("layer_b found nothing");
                candidates = Vec::new();
            }
            Err(e) => {
                warn!(error = %e, "layer_b failed");
                return Err(e.into());
            }
        }
    } else {
        info!("layer_a found nothing; layer_b skipped (no web-search clients configured)");
        candidates = Vec::new();
    }

    // Verification pass — runs for both Layer A and Layer B candidates so
    // false-positive self-citations (e.g. `langchain` linked from the
    // paper body) get flagged too. Parses paper metadata once and reuses
    // it across candidates.
    let paper = if web_clients.is_some() {
        extract_front_matter(paper_markdown)
    } else {
        None
    };
    if !candidates.is_empty() {
        match web_clients {
            Some(clients) => {
                run_verification(&mut candidates, paper.as_ref(), clients).await;
            }
            None => {
                debug!("verifier disabled (no web-search clients); skipping verification pass");
            }
        }
    }

    // Fallback: Layer A's PDF links sometimes point at unrelated repos
    // (awesome-lists, baselines, dependencies). If the verifier didn't
    // confirm any candidate as official/third-party, try web search once
    // as a fallback. Gated on:
    //   - `layer_a_had_results` so we don't re-run Layer B when the
    //     no-Layer-A branch above already executed it (would loop on
    //     low-confidence web results).
    //   - `web_clients` available.
    //   - At least one Layer A candidate was actually verified (no
    //     verification → we don't know if it's bad, so don't burn search
    //     budget on it).
    if layer_a_had_results && !layer_b_attempted && needs_web_search_fallback(&candidates) {
        if let Some(clients) = web_clients {
            info!("all layer_a candidates verified suspect; trying layer_b as fallback");
            layer_b_attempted = true;
            match run_layer_b(paper_markdown, clients).await {
                Ok(Some(mut c)) => {
                    run_verification(std::slice::from_mut(&mut c), paper.as_ref(), clients).await;
                    info!(url = %c.url, "layer_b fallback found a candidate");
                    candidates.push(c);
                }
                Ok(None) => {
                    info!("layer_b fallback found nothing");
                }
                Err(e) => {
                    // Non-fatal: keep the Layer A candidates we already have.
                    warn!(error = %e, "layer_b fallback failed");
                }
            }
        }
    }

    Ok(DetectionOutcome {
        candidates,
        layer_b_attempted,
    })
}

/// Decide whether to run Layer B as a fallback after verifying Layer A's
/// candidates. True iff no verified candidate is `Official` — i.e. we
/// haven't found the authors' own repo. We still run fallback when
/// `third_party` reimplementations are present, because:
/// (a) workspaces with `accept_unofficial_implementations=false` drop
///     `third_party` rows, so leaving them as the only result yields an
///     empty detection;
/// (b) the verifier's verdict isn't always stable — a small-model run
///     might call the same repo `third_party` one time and `inconclusive`
///     the next. Always seeking an `Official` complement is robust to
///     that wobble.
///
/// Unverified candidates don't count toward the decision — missing
/// verification means we don't *know* what we have, so we don't burn
/// a web-search round-trip on speculation.
fn needs_web_search_fallback(candidates: &[RepoCandidate]) -> bool {
    let mut saw_verified = false;
    for c in candidates {
        if let Some(v) = c.verification.as_ref() {
            saw_verified = true;
            if matches!(v.verdict, VerificationVerdict::Official) {
                return false;
            }
        }
    }
    saw_verified
}

/// Apply the verifier to each candidate in place. Verifier failures are
/// logged and swallowed — the candidate persists without verification.
async fn run_verification(
    candidates: &mut [RepoCandidate],
    paper: Option<&PaperFrontMatter>,
    clients: &WebSearchClients,
) {
    let Some(paper) = paper else {
        debug!("verifier: no front-matter extracted, skipping verification");
        return;
    };
    for c in candidates.iter_mut() {
        match verify_candidate(c, paper, &clients.crawl4ai, Arc::clone(&clients.llm)).await {
            Ok(report) => {
                info!(
                    url = %c.url,
                    verdict = report.verdict.as_str(),
                    confidence = report.confidence,
                    "verifier report"
                );
                // Derive confidence from the verifier's verdict. Layer A
                // assigned `High` unconditionally and Layer B from search
                // rank — the verdict is a stronger signal than either, so
                // it wins. Mapping:
                //   official     → High   (author-owned repo)
                //   third_party  → Medium (faithful reimpl, but not authors')
                //   inconclusive → Medium (README ambiguous; needs review)
                //   unrelated    → Low    (langchain/awesome-list/etc.)
                c.confidence = match report.verdict {
                    VerificationVerdict::Official => Confidence::High,
                    VerificationVerdict::ThirdParty | VerificationVerdict::Inconclusive => {
                        Confidence::Medium
                    }
                    VerificationVerdict::Unrelated => Confidence::Low,
                };
                c.verification = Some(report);
            }
            Err(e) => {
                warn!(url = %c.url, error = %e, "verifier skipped — candidate persists without verification");
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct DetectionOutcome {
    pub candidates: Vec<RepoCandidate>,
    pub layer_b_attempted: bool,
}

impl DetectionOutcome {
    pub fn layer_a_count(&self) -> usize {
        self.candidates
            .iter()
            .filter(|c| matches!(c.detection_method, DetectionMethod::PdfLink))
            .count()
    }

    /// Count of candidates from any Layer B/C source (GitHub API or
    /// SearXNG/Crawl4AI). Naming stays "layer_b" for backwards-compat
    /// with the persisted `document_repo_detections.layer_b_candidates`
    /// column.
    pub fn layer_b_count(&self) -> usize {
        self.candidates
            .iter()
            .filter(|c| {
                matches!(
                    c.detection_method,
                    DetectionMethod::GithubApi | DetectionMethod::WebSearch
                )
            })
            .count()
    }
}

async fn run_layer_a(pdf_bytes: &[u8]) -> Result<Vec<RepoCandidate>, RepoDetectionError> {
    // pdfium is not async-safe; pay spawn_blocking here.
    let bytes = pdf_bytes.to_vec();
    let result = tokio::task::spawn_blocking(move || {
        let extraction = edgequake_pdf::extract_links(&bytes)
            .map_err(|e| RepoDetectionError::PdfParse(e.to_string()))?;
        Ok::<_, RepoDetectionError>(edgequake_pdf::detect_repos(&extraction))
    })
    .await
    .map_err(|e| RepoDetectionError::PdfParse(format!("spawn_blocking join: {e}")))??;

    Ok(result
        .into_iter()
        .map(|r| RepoCandidate {
            host: RepoHost::from_pdf(r.host),
            owner: r.owner,
            repo: r.repo,
            url: r.url,
            detection_method: DetectionMethod::PdfLink,
            pdf_page_index: Some(r.page_index as i32),
            search_rank: None,
            source_url: None,
            // PDF link in the body = high-confidence self-citation.
            confidence: Confidence::High,
            verification: None,
        })
        .collect())
}

async fn run_layer_b(
    paper_markdown: &str,
    clients: &WebSearchClients,
) -> Result<Option<RepoCandidate>, RepoResolverError> {
    let cfg = RepoResolverConfig::default();
    match resolve_repo(
        paper_markdown,
        &clients.github,
        &clients.searxng,
        &clients.crawl4ai,
        Arc::clone(&clients.llm),
        &cfg,
    )
    .await
    {
        Ok(r) => {
            // Rank 0 = high confidence, rank 1-2 = medium, deeper = low.
            let confidence = match r.source_rank {
                Some(0) => Confidence::High,
                Some(1 | 2) => Confidence::Medium,
                _ => Confidence::Low,
            };
            // Map the resolver's per-source tag onto the persisted
            // detection_method column. The schema's CHECK constraint
            // (migration 054) accepts both `github_api` and
            // `web_search`.
            let detection_method = match r.source {
                HitSource::GithubApi => DetectionMethod::GithubApi,
                HitSource::WebSearch => DetectionMethod::WebSearch,
            };
            Ok(Some(RepoCandidate {
                host: RepoHost::from_pdf(r.host),
                owner: r.owner,
                repo: r.repo,
                url: r.url,
                detection_method,
                pdf_page_index: None,
                search_rank: r.source_rank.map(|r| r as i32),
                source_url: r.source_url,
                confidence,
                verification: None,
            }))
        }
        // NotFound / NoFrontMatter are non-errors — just "nothing to return".
        Err(RepoResolverError::NotFound | RepoResolverError::NoFrontMatter) => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_detection::verify::VerificationReport;

    #[tokio::test]
    async fn layer_b_skipped_when_no_clients_configured() {
        let outcome = run_detection(
            &[], // no pdf → layer A returns empty
            "# Title\n\nAlice Smith, Bob Jones\n",
            None,
            &RepoDetectionConfig::default(),
        )
        .await
        .unwrap();
        assert!(outcome.candidates.is_empty());
        assert!(!outcome.layer_b_attempted);
    }

    fn cand_with_verdict(url: &str, verdict: Option<VerificationVerdict>) -> RepoCandidate {
        RepoCandidate {
            host: RepoHost::Github,
            owner: "owner".into(),
            repo: "repo".into(),
            url: url.into(),
            detection_method: DetectionMethod::PdfLink,
            pdf_page_index: Some(0),
            search_rank: None,
            source_url: None,
            confidence: Confidence::High,
            verification: verdict.map(|v| VerificationReport {
                verdict: v,
                confidence: 0.5,
                rationale: "test".into(),
            }),
        }
    }

    #[test]
    fn fallback_skipped_when_unverified() {
        // No verifier ran — we don't know if Layer A is good or bad, so
        // don't burn a web-search round-trip.
        let cands = vec![cand_with_verdict("https://github.com/a/b", None)];
        assert!(!needs_web_search_fallback(&cands));
    }

    #[test]
    fn fallback_skipped_when_any_official() {
        let cands = vec![
            cand_with_verdict("https://github.com/a/b", Some(VerificationVerdict::Unrelated)),
            cand_with_verdict("https://github.com/c/d", Some(VerificationVerdict::Official)),
        ];
        assert!(!needs_web_search_fallback(&cands));
    }

    #[test]
    fn fallback_triggered_when_only_third_party() {
        // Workspace filter drops third_party rows when accept_unofficial
        // is false, so leaving them as the only verified candidate
        // produces an empty detection. We still web-search for an
        // Official complement.
        let cands = vec![
            cand_with_verdict(
                "https://github.com/a/b",
                Some(VerificationVerdict::Unrelated),
            ),
            cand_with_verdict(
                "https://github.com/c/d",
                Some(VerificationVerdict::ThirdParty),
            ),
        ];
        assert!(needs_web_search_fallback(&cands));
    }

    #[test]
    fn fallback_triggered_when_all_unrelated_or_inconclusive() {
        let cands = vec![
            cand_with_verdict(
                "https://github.com/a/b",
                Some(VerificationVerdict::Unrelated),
            ),
            cand_with_verdict(
                "https://github.com/c/d",
                Some(VerificationVerdict::Inconclusive),
            ),
        ];
        assert!(needs_web_search_fallback(&cands));
    }

    #[test]
    fn fallback_skipped_when_no_candidates_verified() {
        // Mixed verified + unverified, but no verified candidate at all.
        // (Defensive case: empty list should also return false.)
        let cands: Vec<RepoCandidate> = vec![];
        assert!(!needs_web_search_fallback(&cands));
    }
}
