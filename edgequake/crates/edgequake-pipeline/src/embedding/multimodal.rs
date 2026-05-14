//! Multimodal embedding HTTP client.
//!
//! Calls the Jina `feat-v5-omni` llama.cpp fork's `/v1/embeddings` endpoint
//! with text in `input` and image data URIs in a top-level `images` array
//! (the fork's actual accepted shape — OpenAI's chat-completions
//! content-parts shape is rejected with a `"prompt" elements must be...`
//! HTTP 500). Served via llama-swap (see
//! `~/infra/dashboard/llama-swap/config.yaml`).
//!
//! ## Asymmetric retrieval
//!
//! The model is trained with literal `Query: ` / `Document: ` prefixes — the
//! adapter offsets the two sides in vector space. We prepend client-side
//! because neither vLLM nor llama.cpp's `/v1/embeddings` accepts a
//! `prompt_name` field. See [`EmbeddingRole`].
//!
//! ## Multimodal fusion
//!
//! Figure chunks embed as **one** input combining the caption text and the
//! image bytes — not two separate vectors averaged. Jina v5 omni's retrieval
//! projector was trained with image + surrounding text on the document side
//! of the InfoNCE pair, so fused input matches the training distribution.

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Asymmetric retrieval role. Jina v5 omni's retrieval adapter offsets queries
/// and documents into different subspaces via a literal prefix. Use [`Query`]
/// at search time and [`Document`] for everything indexed.
///
/// [`Query`]: EmbeddingRole::Query
/// [`Document`]: EmbeddingRole::Document
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingRole {
    Query,
    Document,
}

impl EmbeddingRole {
    fn prefix(self) -> &'static str {
        match self {
            EmbeddingRole::Query => "Query: ",
            EmbeddingRole::Document => "Document: ",
        }
    }
}

/// Input to a single embedding call.
#[derive(Debug, Clone)]
pub enum EmbeddingInput<'a> {
    /// Plain text. The client prepends the [`EmbeddingRole`] prefix before
    /// sending; the caller passes the raw text.
    Text(&'a str),
    /// Caption + image bytes fused into one multimodal input. The caption gets
    /// the role prefix; the image is sent as a base64 data URI in the same
    /// content-parts array.
    Figure {
        caption: &'a str,
        bytes: &'a [u8],
        mime: &'a str,
    },
}

#[derive(Debug, Error)]
pub enum MultimodalEmbeddingError {
    #[error("http transport: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("server returned HTTP {status}: {body}")]
    BadStatus { status: u16, body: String },
    #[error("response missing embedding data")]
    EmptyResponse,
    #[error("response embedding length 0")]
    EmptyVector,
    #[error("invalid response shape: {0}")]
    InvalidShape(String),
}

/// HTTP client for the multimodal-embeddings endpoint.
///
/// One instance per (base_url, model). Cheap to clone — the inner `reqwest`
/// client pools connections so callers should share a single instance per
/// process and clone it into tasks rather than constructing per call.
#[derive(Debug, Clone)]
pub struct MultimodalEmbeddingClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
}

impl MultimodalEmbeddingClient {
    /// Construct a client. `base_url` is the OpenAI-compat root (the
    /// `/v1/embeddings` suffix is appended internally) — e.g.
    /// `"http://localhost:8060/v1"` for llama-swap.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let http = reqwest::Client::builder()
            // Image payloads can be tens of MB worth of base64; keep timeout
            // generous. llama-swap also cold-spawns the backend container on
            // the first request, which can take a few seconds.
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("reqwest::Client::new must succeed with default config");
        Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
        }
    }

    /// Construct from env vars — kept around for live smoke tests and
    /// debugging. Production code paths read workspace settings instead and
    /// call `Self::new` with the resolved provider URL + model name.
    ///
    /// `EDGEQUAKE_JINA_OMNI_URL`   — base URL (default
    ///                              `http://localhost:8060/v1`)
    /// `EDGEQUAKE_JINA_OMNI_MODEL` — model id (default
    ///                              `jina-embeddings-v5-omni-small-retrieval`)
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("EDGEQUAKE_JINA_OMNI_URL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "http://localhost:8060/v1".to_string());
        let model = std::env::var("EDGEQUAKE_JINA_OMNI_MODEL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "jina-embeddings-v5-omni-small-retrieval".to_string());
        // Env-only construction is always valid — the empty URL fallback
        // above means we always have something to talk to. Callers that want
        // strict opt-in should branch on the raw env var themselves.
        Some(Self::new(url, model))
    }

    /// Embed a single input.
    pub async fn embed(
        &self,
        input: EmbeddingInput<'_>,
        role: EmbeddingRole,
    ) -> Result<Vec<f32>, MultimodalEmbeddingError> {
        let body = self.build_request_body(&input, role);
        let url = format!("{}/embeddings", self.base_url);
        let resp = self.http.post(&url).json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(MultimodalEmbeddingError::BadStatus {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: EmbeddingResponse = resp.json().await?;
        let first = parsed
            .data
            .into_iter()
            .next()
            .ok_or(MultimodalEmbeddingError::EmptyResponse)?;
        if first.embedding.is_empty() {
            return Err(MultimodalEmbeddingError::EmptyVector);
        }
        Ok(first.embedding)
    }

    /// The model id this client targets, exposed so callers can log/audit
    /// which embedding produced a given vector.
    pub fn model(&self) -> &str {
        &self.model
    }

    fn build_request_body(
        &self,
        input: &EmbeddingInput<'_>,
        role: EmbeddingRole,
    ) -> serde_json::Value {
        match input {
            // Plain text → string input. Simplest path, matches what
            // `curl -d '{"input": "..."}'` would send.
            EmbeddingInput::Text(text) => {
                let prefixed = format!("{}{}", role.prefix(), text);
                serde_json::json!({
                    "model": self.model,
                    "input": prefixed,
                })
            }
            // Figure → caption in `input` (as a single string with the role
            // prefix) and the image as a base64 data URI in a top-level
            // `images` array. This is the shape the Jina v5-omni llama.cpp
            // fork's `/v1/embeddings` accepts; the OpenAI chat-completions
            // content-parts shape (`input: [[{type,text,image_url}]]`) is
            // rejected with `"prompt" elements must be a string, a list of
            // tokens, ...` HTTP 500.
            EmbeddingInput::Figure {
                caption,
                bytes,
                mime,
            } => {
                let prefixed = format!("{}{}", role.prefix(), caption);
                let data_uri = format!("data:{};base64,{}", mime, B64.encode(bytes));
                serde_json::json!({
                    "model": self.model,
                    "input": prefixed,
                    "images": [data_uri],
                })
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingEntry>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingEntry {
    embedding: Vec<f32>,
}

/// Used internally by the JSON body builder when we want to round-trip the
/// content-parts shape through serde for tests.
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
struct ImageUrl {
    url: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_prefixes_match_jina_training_distribution() {
        assert_eq!(EmbeddingRole::Query.prefix(), "Query: ");
        assert_eq!(EmbeddingRole::Document.prefix(), "Document: ");
    }

    #[test]
    fn text_input_serializes_as_plain_string_with_prefix() {
        let c = MultimodalEmbeddingClient::new(
            "http://example",
            "jina-embeddings-v5-omni-small-retrieval",
        );
        let body = c.build_request_body(&EmbeddingInput::Text("hello"), EmbeddingRole::Document);
        let json = serde_json::to_string(&body).unwrap();
        // Model field present.
        assert!(json.contains("\"model\":\"jina-embeddings-v5-omni-small-retrieval\""));
        // Input is a plain string, not an array, and carries the prefix.
        assert!(json.contains("\"input\":\"Document: hello\""));
    }

    #[test]
    fn query_role_uses_query_prefix() {
        let c = MultimodalEmbeddingClient::new("http://example", "m");
        let body = c.build_request_body(&EmbeddingInput::Text("what is X?"), EmbeddingRole::Query);
        assert!(body["input"]
            .as_str()
            .unwrap()
            .starts_with("Query: what is X?"));
    }

    #[test]
    fn figure_input_serializes_with_input_string_and_images_array() {
        let c = MultimodalEmbeddingClient::new("http://example", "m");
        let bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR";
        let body = c.build_request_body(
            &EmbeddingInput::Figure {
                caption: "Figure 2: System diagram.",
                bytes,
                mime: "image/png",
            },
            EmbeddingRole::Document,
        );
        // `input` is the prefixed caption as a plain string — same shape the
        // text-only path uses. The Jina v5-omni llama.cpp fork rejects the
        // OpenAI content-parts array shape with HTTP 500.
        assert_eq!(
            body["input"].as_str().unwrap(),
            "Document: Figure 2: System diagram."
        );
        // `images` is a top-level array of data URIs, one per image.
        let images = body.get("images").unwrap().as_array().unwrap();
        assert_eq!(images.len(), 1, "expected exactly one image");
        let url = images[0].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
        // Round-trip the base64 chunk and verify byte-equality.
        let b64 = url.trim_start_matches("data:image/png;base64,");
        let decoded = B64.decode(b64).unwrap();
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn from_env_falls_back_to_llama_swap_default() {
        // Strip env so we exercise the fallback path. SAFETY: tests are
        // single-threaded by default; this is the only knob we touch.
        unsafe {
            std::env::remove_var("EDGEQUAKE_JINA_OMNI_URL");
            std::env::remove_var("EDGEQUAKE_JINA_OMNI_MODEL");
        }
        let c = MultimodalEmbeddingClient::from_env().unwrap();
        assert_eq!(c.base_url, "http://localhost:8060/v1");
        assert_eq!(c.model, "jina-embeddings-v5-omni-small-retrieval");
    }

    /// Live-server smoke test. Talks to the actual llama-swap on port 8060
    /// (cold-spawning the container if needed). Ignored by default — run
    /// explicitly with:
    ///
    /// ```sh
    /// cargo test -p edgequake-pipeline --lib \
    ///     embedding::multimodal::tests::live_smoke -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore = "requires llama-swap with jina-embeddings-v5-omni-small-retrieval"]
    async fn live_smoke() {
        let c = MultimodalEmbeddingClient::from_env().unwrap();
        // Text path.
        let q = c
            .embed(EmbeddingInput::Text("hello world"), EmbeddingRole::Query)
            .await
            .unwrap();
        assert_eq!(q.len(), 1024);
        let d = c
            .embed(
                EmbeddingInput::Text("hello world"),
                EmbeddingRole::Document,
            )
            .await
            .unwrap();
        assert_eq!(d.len(), 1024);
        // Asymmetric prefix really lands different points in the space.
        let cos: f32 = q.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
        eprintln!("cos(query vs doc, same text) = {cos:.4}");
        assert!(
            cos < 0.999,
            "query and document prefixes should land different vectors"
        );
    }
}
