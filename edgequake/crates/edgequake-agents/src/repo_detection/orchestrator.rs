//! Top-level repo-detection flow: Layer A (PDF annotations) → Layer B
//! (web-search fallback) → ranked candidates ready for persistence.
//!
//! Returns the raw candidate list; the caller (API task processor) handles
//! DB writes, status transitions, and confidence-based filtering.

use std::sync::Arc;

use edgequake_llm::traits::LLMProvider;
use tracing::{debug, info, warn};

use super::types::{Confidence, DetectionMethod, RepoCandidate, RepoHost};
use crate::web_search::{
    resolve_repo, Crawl4aiClient, RepoResolverConfig, RepoResolverError, SearxngClient,
};

/// External dependencies for Layer B. If `None`, Layer B is skipped silently
/// (so EdgeQuake deployments without SearXNG/Crawl4AI still get Layer A).
pub struct WebSearchClients {
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

/// Run Layer A against `pdf_bytes`. If it finds anything, return; otherwise
/// fall through to Layer B using `paper_markdown` + `web_clients`.
///
/// Never fails if Layer A returned at least one candidate, even when Layer B
/// would have failed — partial results are fine.
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
    if !layer_a.is_empty() {
        info!(count = layer_a.len(), "layer_a found reference repos");
        return Ok(DetectionOutcome {
            candidates: layer_a,
            layer_b_attempted: false,
        });
    }

    let Some(clients) = web_clients else {
        info!("layer_a found nothing; layer_b skipped (no web-search clients configured)");
        return Ok(DetectionOutcome::default());
    };

    debug!("layer_a empty; falling through to layer_b");
    match run_layer_b(paper_markdown, clients).await {
        Ok(Some(c)) => {
            info!("layer_b found a reference repo");
            Ok(DetectionOutcome {
                candidates: vec![c],
                layer_b_attempted: true,
            })
        }
        Ok(None) => {
            info!("layer_b found nothing");
            Ok(DetectionOutcome {
                candidates: Vec::new(),
                layer_b_attempted: true,
            })
        }
        Err(e) => {
            warn!(error = %e, "layer_b failed");
            Err(e.into())
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

    pub fn layer_b_count(&self) -> usize {
        self.candidates
            .iter()
            .filter(|c| matches!(c.detection_method, DetectionMethod::WebSearch))
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
            Ok(Some(RepoCandidate {
                host: RepoHost::from_pdf(r.host),
                owner: r.owner,
                repo: r.repo,
                url: r.url,
                detection_method: DetectionMethod::WebSearch,
                pdf_page_index: None,
                search_rank: r.source_rank.map(|r| r as i32),
                source_url: r.source_url,
                confidence,
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
}
