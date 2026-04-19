//! Thin JSON client for a self-hosted SearXNG instance.
//!
//! Requires `format=json` to be enabled in SearXNG's `settings.yml`
//! (`search.formats: [html, json]`). The user confirmed their deployment at
//! `http://localhost:8888` already serves JSON.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SearxngError {
    #[error("HTTP request to SearXNG failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("SearXNG returned non-2xx status {status}: {body}")]
    Status {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("failed to parse SearXNG JSON response: {0}")]
    Parse(#[from] serde_json::Error),
}

/// One result item from SearXNG's `format=json` payload. Fields other than
/// `url`/`title`/`content` are intentionally discarded — they vary by engine
/// and we don't need them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearxngResult {
    pub url: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct SearxngResponse {
    #[serde(default)]
    results: Vec<SearxngResult>,
}

#[derive(Debug, Clone)]
pub struct SearxngClient {
    base_url: String,
    http: reqwest::Client,
}

impl SearxngClient {
    /// Build a client pointing at e.g. `http://localhost:8888`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_http_client(base_url, reqwest::Client::new())
    }

    pub fn with_http_client(base_url: impl Into<String>, http: reqwest::Client) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http,
        }
    }

    /// Issue a search and return the raw result list (in engine-ranked order).
    pub async fn search(&self, query: &str) -> Result<Vec<SearxngResult>, SearxngError> {
        let url = format!("{}/search", self.base_url);
        let resp = self
            .http
            .get(&url)
            .query(&[("q", query), ("format", "json")])
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(SearxngError::Status {
                status,
                body: truncate(&text, 400),
            });
        }
        let parsed: SearxngResponse = serde_json::from_str(&text)?;
        Ok(parsed.results)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut cut = max;
        while !s.is_char_boundary(cut) && cut > 0 {
            cut -= 1;
        }
        format!("{}…", &s[..cut])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_minimal_response() {
        let body = r#"{"results":[{"url":"https://example.com","title":"x","content":"y"}]}"#;
        let parsed: SearxngResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.results.len(), 1);
        assert_eq!(parsed.results[0].url, "https://example.com");
    }

    #[test]
    fn deserialises_empty() {
        let body = r#"{"results":[]}"#;
        let parsed: SearxngResponse = serde_json::from_str(body).unwrap();
        assert!(parsed.results.is_empty());
    }

    #[test]
    fn tolerates_missing_title_content() {
        let body = r#"{"results":[{"url":"https://example.com"}]}"#;
        let parsed: SearxngResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.results[0].title, "");
        assert_eq!(parsed.results[0].content, "");
    }
}
