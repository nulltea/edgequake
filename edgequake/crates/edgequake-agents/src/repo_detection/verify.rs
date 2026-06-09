//! Post-detection verifier: scores each candidate repo against paper
//! metadata + the repo's README via the workspace LLM, and returns a
//! verdict. Does NOT auto-reject or delete — verdicts are persisted
//! alongside the candidate so a reviewer can see them in the UI.
//!
//! Mirrors the `CodeArtifact.match_rationale` pattern in
//! `code_analysis::types` so the two enrichment steps feel uniform.
//!
//! Failure is non-fatal: README fetch errors, LLM errors, or JSON-parse
//! errors produce either `Inconclusive` + rationale, or `Err(_)` that the
//! caller maps to "no verification for this candidate".

use std::sync::Arc;

use edgequake_llm::traits::LLMProvider;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

use super::types::RepoCandidate;
use crate::web_search::{Crawl4aiClient, PaperFrontMatter};

/// Verifier verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationVerdict {
    /// Repository is the paper's own release (authored by the paper authors).
    Official,
    /// Repository is a faithful third-party reimplementation. Its README
    /// cites the paper by title, arXiv id, or author names.
    ThirdParty,
    /// Repository is unrelated to the paper's topic — e.g. a generic
    /// library the paper happens to link to (`langchain`, `pytorch`).
    Unrelated,
    /// LLM could not decide (returned garbage, abstained, or signaled
    /// uncertainty).
    Inconclusive,
}

impl VerificationVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            VerificationVerdict::Official => "official",
            VerificationVerdict::ThirdParty => "third_party",
            VerificationVerdict::Unrelated => "unrelated",
            VerificationVerdict::Inconclusive => "inconclusive",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "official" => Some(Self::Official),
            "third_party" => Some(Self::ThirdParty),
            "unrelated" => Some(Self::Unrelated),
            "inconclusive" => Some(Self::Inconclusive),
            _ => None,
        }
    }
}

/// Scored verification attached to a [`RepoCandidate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub verdict: VerificationVerdict,
    /// LLM-reported confidence in `[0, 1]`.
    pub confidence: f32,
    pub rationale: String,
}

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("LLM call failed: {0}")]
    Llm(String),
}

/// Max README chars to include in the verifier prompt. README + paper
/// metadata together should fit comfortably in a 16k-token context.
const MAX_README_CHARS: usize = 4_000;
/// Max README chars the `Crawl4aiClient` returns before we clip. Separate
/// from prompt-level clipping so we can reason about crawl cost.
const CRAWL_READ_CAP: usize = 16_000;

/// Verify a single candidate. Best-effort — errors are logged, the caller
/// receives a result.
pub async fn verify_candidate(
    candidate: &RepoCandidate,
    paper: &PaperFrontMatter,
    crawl: &Crawl4aiClient,
    llm: Arc<dyn LLMProvider>,
) -> Result<VerificationReport, VerifyError> {
    // 1. Fetch README. Prefer raw.githubusercontent.com since it returns
    //    plain markdown without any site chrome; fall back to the repo's
    //    github.com page (whose rendered README Crawl4AI will extract).
    let readme_excerpt = fetch_readme(crawl, candidate).await;

    if readme_excerpt.is_empty() {
        debug!(
            url = %candidate.url,
            "verifier: README unreachable; LLM will score on metadata only"
        );
    }

    // 2. Build prompt and call LLM.
    let prompt = build_prompt(candidate, paper, &readme_excerpt);
    let resp = llm
        .complete(&prompt)
        .await
        .map_err(|e| VerifyError::Llm(e.to_string()))?;
    let raw = resp.content.trim();

    // 3. Parse structured output. Robust to surrounding prose: grab the
    //    first `{...}` block and try to parse it.
    Ok(parse_verdict(raw))
}

async fn fetch_readme(crawl: &Crawl4aiClient, c: &RepoCandidate) -> String {
    // Only GitHub has a well-known raw URL; others we fall back to the
    // canonical URL and let Crawl4AI render.
    let raw_candidates: Vec<String> = if matches!(c.host, super::types::RepoHost::Github) {
        vec![
            format!(
                "https://raw.githubusercontent.com/{}/{}/HEAD/README.md",
                c.owner, c.repo
            ),
            format!(
                "https://raw.githubusercontent.com/{}/{}/HEAD/readme.md",
                c.owner, c.repo
            ),
            format!(
                "https://raw.githubusercontent.com/{}/{}/HEAD/README.rst",
                c.owner, c.repo
            ),
            c.url.clone(),
        ]
    } else {
        vec![c.url.clone()]
    };

    for url in &raw_candidates {
        match crawl.markdown(url).await {
            Ok(r) if r.success && !r.markdown.is_empty() => {
                return r.markdown.chars().take(CRAWL_READ_CAP).collect();
            }
            Ok(_) => continue,
            Err(e) => {
                warn!(url = %url, error = %e, "verifier: README fetch failed");
                continue;
            }
        }
    }
    String::new()
}

fn build_prompt(c: &RepoCandidate, p: &PaperFrontMatter, readme: &str) -> String {
    let authors = if p.authors.is_empty() {
        p.first_author.clone().unwrap_or_default()
    } else {
        p.authors.join(", ")
    };
    let abstract_line = p
        .abstract_excerpt
        .as_deref()
        .unwrap_or("(no abstract available)");
    let readme_clipped: String = readme.chars().take(MAX_README_CHARS).collect();
    let readme_section = if readme_clipped.is_empty() {
        "(README unavailable — score using URL + metadata only, and lean \
         toward `inconclusive` or `unrelated` unless the URL itself is a \
         clear match.)"
            .to_string()
    } else {
        format!("--- README (clipped) ---\n{readme_clipped}")
    };

    format!(
        "You are verifying whether a GitHub/GitLab/Bitbucket repository implements a specific research paper.\n\
         \n\
         Classify the relationship with ONE of:\n\
         - `official`      : The repo is the paper authors' OWN release. Any ONE of these is sufficient:\n\
         \x20    * the repo owner is an author, OR the authors' institution / lab / research group — these are often organisation accounts (e.g. a university-lab GitHub org whose name is an acronym or institution name, not a personal handle);\n\
         \x20    * the README presents the repo AS this paper's own implementation, prototype, artifact, or released code — phrasings like \"this repository is the/a prototype of <paper>\", \"code for our paper\", \"official implementation\", \"we release\" — without disclaiming authorship.\n\
         \x20    A repo whose name matches the paper's system/method name and is owned by a research-group organisation is a STRONG `official` signal.\n\
         - `third_party`   : A reimplementation by someone OTHER than the authors. The README must explicitly frame it that way — e.g. \"unofficial\", \"reproduction\", \"re-implementation\", \"my implementation of\", \"reproduce the results of\". IMPORTANT: merely citing the paper's title / arXiv id / authors is NOT enough to choose `third_party` — the authors' own repo cites its paper too. If the README cites or describes the paper but does not disclaim authorship, prefer `official`.\n\
         - `unrelated`     : The repo is a generic library, dataset, baseline, or tooling that the paper merely uses/cites (examples: langchain, pytorch, numpy, transformers, scikit-learn). Also applies to author profile pages with no specific repo, and to `awesome-*` / paper-list aggregators.\n\
         - `inconclusive`  : Not enough evidence to decide.\n\
         \n\
         --- PAPER ---\n\
         Title:   {title}\n\
         Authors: {authors}\n\
         Abstract: {abstract_line}\n\
         \n\
         --- REPOSITORY ---\n\
         URL:   {url}\n\
         Owner: {owner}\n\
         Repo:  {repo}\n\
         \n\
         {readme_section}\n\
         \n\
         --- INSTRUCTIONS ---\n\
         Reply with ONLY a single JSON object on one line, no code fences, no prose before or after:\n\
         {{\"verdict\": \"official|third_party|unrelated|inconclusive\", \"confidence\": <0.0-1.0>, \"rationale\": \"one-sentence reason citing README evidence or URL signals\"}}\n",
        title = p.title,
        authors = authors,
        abstract_line = abstract_line,
        url = c.url,
        owner = c.owner,
        repo = c.repo,
        readme_section = readme_section,
    )
}

/// Parse the LLM output, tolerating leading/trailing chatter by grabbing
/// the first balanced `{...}` block. Always returns a report — on any
/// parse failure the verdict is `Inconclusive`.
fn parse_verdict(raw: &str) -> VerificationReport {
    let Some(json_slice) = extract_first_json_object(raw) else {
        return inconclusive("verifier returned no JSON object");
    };

    #[derive(Deserialize)]
    struct Parsed {
        verdict: Option<String>,
        confidence: Option<f32>,
        rationale: Option<String>,
    }

    let parsed: Parsed = match serde_json::from_str(&json_slice) {
        Ok(p) => p,
        Err(e) => {
            debug!(error = %e, raw = %json_slice, "verifier JSON parse failed");
            return inconclusive("verifier returned unparseable JSON");
        }
    };

    let verdict = parsed
        .verdict
        .as_deref()
        .and_then(VerificationVerdict::parse)
        .unwrap_or(VerificationVerdict::Inconclusive);
    let confidence = parsed.confidence.unwrap_or(0.0).clamp(0.0, 1.0);
    let rationale = parsed
        .rationale
        .unwrap_or_else(|| "verifier returned no rationale".to_string());

    VerificationReport {
        verdict,
        confidence,
        rationale,
    }
}

fn inconclusive(reason: &str) -> VerificationReport {
    VerificationReport {
        verdict: VerificationVerdict::Inconclusive,
        confidence: 0.0,
        rationale: reason.to_string(),
    }
}

/// Find the first balanced `{...}` block in `s` and return it as an owned
/// string. Handles nested braces and strings with escaped quotes. Returns
/// `None` if no complete object is present.
fn extract_first_json_object(s: &str) -> Option<String> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escape {
                escape = false;
                continue;
            }
            if b == b'\\' {
                escape = true;
                continue;
            }
            if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain_json() {
        let raw = r#"{"verdict":"official","confidence":0.9,"rationale":"author owns repo"}"#;
        let r = parse_verdict(raw);
        assert_eq!(r.verdict, VerificationVerdict::Official);
        assert!((r.confidence - 0.9).abs() < 1e-6);
        assert!(r.rationale.contains("author"));
    }

    #[test]
    fn parse_with_chatter_around_json() {
        let raw = r#"Sure, here is the JSON:
```json
{"verdict": "third_party", "confidence": 0.65, "rationale": "README cites arXiv:1234.5678"}
```
Hope this helps!"#;
        let r = parse_verdict(raw);
        assert_eq!(r.verdict, VerificationVerdict::ThirdParty);
    }

    #[test]
    fn parse_clamps_confidence() {
        let raw = r#"{"verdict":"unrelated","confidence":2.5,"rationale":"langchain"}"#;
        let r = parse_verdict(raw);
        assert!(r.confidence <= 1.0);
    }

    #[test]
    fn parse_garbage_returns_inconclusive() {
        let r = parse_verdict("this is not json");
        assert_eq!(r.verdict, VerificationVerdict::Inconclusive);
    }

    #[test]
    fn parse_unknown_verdict_string() {
        let raw = r#"{"verdict":"maybe","confidence":0.5,"rationale":"idk"}"#;
        let r = parse_verdict(raw);
        assert_eq!(r.verdict, VerificationVerdict::Inconclusive);
    }

    #[test]
    fn extract_handles_nested_and_strings() {
        let s = r#"pre {"a": {"b": "}"}, "c": 1} tail"#;
        let got = extract_first_json_object(s).unwrap();
        assert_eq!(got, r#"{"a": {"b": "}"}, "c": 1}"#);
    }
}
