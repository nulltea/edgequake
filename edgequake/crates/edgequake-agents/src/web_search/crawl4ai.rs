//! Thin JSON client for a self-hosted Crawl4AI instance.
//!
//! We only use `POST /md` which fetches and cleans a single URL to markdown.
//! The user's deployment runs at `http://localhost:11235`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Crawl4aiError {
    #[error("HTTP request to Crawl4AI failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Crawl4AI returned non-2xx status {status}: {body}")]
    Status {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("failed to parse Crawl4AI JSON response: {0}")]
    Parse(#[from] serde_json::Error),
}

/// Subset of the `POST /md` response we care about.
#[derive(Debug, Clone, Deserialize)]
pub struct MarkdownResponse {
    pub url: String,
    pub markdown: String,
    #[serde(default)]
    pub success: bool,
}

#[derive(Debug, Serialize)]
struct MarkdownRequest<'a> {
    url: &'a str,
    /// "raw" = as-crawled, "fit" = relevance-filtered. Raw is enough for us
    /// since we pass the output straight to an LLM that's already robust to noise.
    f: &'a str,
}

#[derive(Debug, Clone)]
pub struct Crawl4aiClient {
    base_url: String,
    http: reqwest::Client,
}

impl Crawl4aiClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_http_client(base_url, reqwest::Client::new())
    }

    pub fn with_http_client(base_url: impl Into<String>, http: reqwest::Client) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http,
        }
    }

    /// Fetch the cleaned markdown representation of `url`.
    pub async fn markdown(&self, url: &str) -> Result<MarkdownResponse, Crawl4aiError> {
        let endpoint = format!("{}/md", self.base_url);
        let body = MarkdownRequest { url, f: "raw" };
        let resp = self.http.post(&endpoint).json(&body).send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(Crawl4aiError::Status {
                status,
                body: truncate(&text, 400),
            });
        }
        let parsed: MarkdownResponse = serde_json::from_str(&text)?;
        Ok(parsed)
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
    fn deserialises_typical_response() {
        let body = r##"{"url":"https://x.com","filter":"raw","query":null,"cache":null,"markdown":"# Hi","success":true}"##;
        let parsed: MarkdownResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.url, "https://x.com");
        assert_eq!(parsed.markdown, "# Hi");
        assert!(parsed.success);
    }
}
