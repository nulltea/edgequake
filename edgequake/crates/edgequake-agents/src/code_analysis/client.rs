//! HTTP client for the `code-analyzer` sidecar.

use std::time::Duration;

use thiserror::Error;

use super::types::{AnalyzerRequest, AnalyzerResponse};

#[derive(Debug, Error)]
pub enum AnalyzerClientError {
    #[error("HTTP error talking to code-analyzer: {0}")]
    Http(#[from] reqwest::Error),
    #[error("code-analyzer returned non-2xx {status}: {body}")]
    Status {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("failed to parse code-analyzer response: {0}")]
    Parse(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct AnalyzerClient {
    base_url: String,
    http: reqwest::Client,
}

impl AnalyzerClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        // Long timeout at the client layer; the server also enforces a
        // per-request `timeout_s` so this is a last-resort backstop.
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(900))
            .build()
            .expect("reqwest client builds");
        Self::with_http_client(base_url, http)
    }

    pub fn with_http_client(base_url: impl Into<String>, http: reqwest::Client) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http,
        }
    }

    pub async fn analyze(
        &self,
        req: &AnalyzerRequest,
    ) -> Result<AnalyzerResponse, AnalyzerClientError> {
        let url = format!("{}/analyze", self.base_url);
        let resp = self.http.post(&url).json(req).send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(AnalyzerClientError::Status {
                status,
                body: truncate(&text, 800),
            });
        }
        Ok(serde_json::from_str(&text)?)
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
