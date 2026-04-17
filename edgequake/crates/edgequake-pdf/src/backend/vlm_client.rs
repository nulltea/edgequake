//! VLM client backend: HTTP client implementing [`RecognitionBackend`] via an
//! OpenAI-compatible `/v1/chat/completions` endpoint (llama.cpp, vLLM, Ollama, etc.).
//!
//! Sends base64-encoded JPEG images alongside task-specific prompts. Designed to
//! work with vision-language models like GLM-OCR, Qwen3-VL, or MiniCPM-V.

use std::io::Cursor;
use std::time::Duration;

use base64::Engine as _;
use image::codecs::jpeg::JpegEncoder;
use image::RgbImage;
use oar_ocr_core::core::OCRError;
use oar_ocr_vl::doc_parser::{RecognitionBackend, RecognitionTask};
use serde_json::{json, Value};
use tracing::{debug, warn};

/// Configuration for the VLM HTTP client.
#[derive(Debug, Clone)]
pub struct VlmClientConfig {
    /// Base URL of the OpenAI-compatible server (e.g. `http://llamacpp:8081`).
    pub base_url: String,
    /// Optional model identifier sent in the API request.
    pub model: Option<String>,
    /// HTTP timeout in seconds (default 120).
    pub timeout_secs: u64,
    /// JPEG quality for image encoding (default 90).
    pub jpeg_quality: u8,
}

impl Default for VlmClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8081".into(),
            model: None,
            timeout_secs: 120,
            jpeg_quality: 90,
        }
    }
}

impl VlmClientConfig {
    /// Build config from environment variables (fallback only).
    ///
    /// Prefer passing base_url and model explicitly from workspace vision settings.
    pub fn from_env() -> Self {
        let base_url = std::env::var("OPENAI_COMPATIBLE_BASE_URL")
            .or_else(|_| std::env::var("OPENAI_BASE_URL"))
            .unwrap_or_else(|_| "http://localhost:8081".into());
        Self {
            base_url,
            ..Default::default()
        }
    }
}

/// HTTP client that implements [`RecognitionBackend`] by calling an OpenAI-compatible
/// vision endpoint. Layout detection still runs locally (PP-DocLayoutV2 ONNX);
/// only recognition is offloaded to the remote VLM.
pub struct VlmClientBackend {
    config: VlmClientConfig,
    agent: ureq::Agent,
}

impl VlmClientBackend {
    pub fn new(config: VlmClientConfig) -> Self {
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(Duration::from_secs(config.timeout_secs)))
                .build(),
        );
        Self { config, agent }
    }

    /// Encode an RgbImage to a base64 JPEG data URI.
    fn encode_image(&self, image: &RgbImage) -> Result<String, OCRError> {
        let mut buf = Cursor::new(Vec::new());
        let encoder = JpegEncoder::new_with_quality(&mut buf, self.config.jpeg_quality);
        image
            .write_with_encoder(encoder)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("VlmClient: JPEG encode failed: {e}"),
            })?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
        Ok(format!("data:image/jpeg;base64,{b64}"))
    }

    /// Call the `/v1/chat/completions` endpoint with an image and prompt.
    fn call_completions(
        &self,
        image: &RgbImage,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String, OCRError> {
        let data_uri = self.encode_image(image)?;

        let base = self.config.base_url.trim_end_matches('/');
        // If the base_url already ends with /v1 (e.g. aperture), don't double it.
        let url = if base.ends_with("/v1") {
            format!("{}/chat/completions", base)
        } else {
            format!("{}/v1/chat/completions", base)
        };

        let mut body = json!({
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": { "url": data_uri }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        });

        if let Some(model) = &self.config.model {
            body["model"] = Value::String(model.clone());
        }

        debug!(
            url = %url,
            prompt = %prompt,
            image_w = image.width(),
            image_h = image.height(),
            "VlmClient: calling completions"
        );

        let body_str = serde_json::to_string(&body).map_err(|e| OCRError::InvalidInput {
            message: format!("VlmClient: failed to serialize request: {e}"),
        })?;

        let mut response = self
            .agent
            .post(&url)
            .content_type("application/json")
            .send(body_str.as_bytes())
            .map_err(|e| OCRError::Inference {
                model_name: "VlmClient".into(),
                context: format!("POST {url}"),
                source: Box::new(e),
            })?;

        let response_str = response
            .body_mut()
            .read_to_string()
            .map_err(|e| OCRError::Inference {
                model_name: "VlmClient".into(),
                context: "read response body".into(),
                source: Box::new(e),
            })?;

        let response_body: Value =
            serde_json::from_str(&response_str).map_err(|e| OCRError::InvalidInput {
                message: format!("VlmClient: invalid JSON response: {e}"),
            })?;

        // Extract choices[0].message.content
        let content = response_body["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| OCRError::InvalidInput {
                message: format!(
                    "VlmClient: unexpected response structure: {}",
                    serde_json::to_string_pretty(&response_body).unwrap_or_default()
                ),
            })?;

        Ok(content.to_string())
    }

    /// Return the prompt for a given recognition task.
    fn prompt_for_task(task: RecognitionTask) -> &'static str {
        match task {
            RecognitionTask::Ocr => {
                "Recognize all text in this image exactly as it appears. Preserve line breaks and formatting."
            }
            RecognitionTask::Table => {
                "Parse the table in this image into HTML. Use <table>, <tr>, <th>, <td> tags."
            }
            RecognitionTask::Formula => {
                "Output the mathematical formula in this image as LaTeX. Only output the LaTeX expression, no explanation."
            }
            RecognitionTask::Chart => {
                "Parse the chart in this image. Use Mermaid format for flowcharts, Markdown for other charts."
            }
        }
    }
}

impl std::fmt::Debug for VlmClientBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VlmClientBackend")
            .field("base_url", &self.config.base_url)
            .field("model", &self.config.model)
            .finish()
    }
}

impl RecognitionBackend for VlmClientBackend {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, OCRError> {
        let prompt = Self::prompt_for_task(task);

        match self.call_completions(&image, prompt, max_tokens) {
            Ok(text) => Ok(text.trim().to_string()),
            Err(e) => {
                warn!(task = ?task, error = %e, "VlmClient: recognition failed");
                Err(e)
            }
        }
    }

    fn needs_table_postprocess(&self) -> bool {
        false // We prompt for HTML directly
    }

    fn needs_formula_preprocess(&self) -> bool {
        false // VLM handles full-region images fine
    }

    fn needs_repetition_truncation(&self) -> bool {
        true // Remote VLMs can produce repetitive output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test: sends /tmp/b2a-page4-4.jpg to a running llama.cpp server.
    ///
    /// Run with: cargo test -p edgequake-pdf -- vlm_client --ignored --nocapture
    /// Requires: llama.cpp server at EDGEQUAKE_LLAMACPP_URL (default localhost:8081)
    ///           and /tmp/b2a-page4-4.jpg (rendered page 4 of B2A.pdf).
    #[test]
    #[ignore] // requires running llama.cpp server
    fn test_recognize_ocr() {
        let img = image::open("/tmp/b2a-page4-4.jpg")
            .expect("open /tmp/b2a-page4-4.jpg")
            .to_rgb8();
        let config = VlmClientConfig::from_env();
        let backend = VlmClientBackend::new(config);
        let result = backend
            .recognize(img, RecognitionTask::Ocr, 4096)
            .expect("recognize failed");
        println!("=== OCR result ({} chars) ===\n{result}", result.len());
        assert!(!result.is_empty());
    }
}
