//! Vision-capable entity/relationship extractor for figure chunks.
//!
//! Mirrors [`SOTAExtractor`](super::SOTAExtractor) — same prompts, same
//! parser, same `LLMProvider::chat` call — but builds a multimodal user
//! message with [`ChatMessage::user_with_images`] so the figure's PNG bytes
//! ride alongside the caption text. `edgequake-llm` v0.5.1 already supports
//! this via `ImageData { data, mime_type, detail }`, so no custom HTTP path
//! is needed.
//!
//! ## Why a vision pass at ingest
//!
//! The workspace-scoped hybrid query path
//! (`edgequake-query::sota_engine::vector_queries::query_hybrid_with_vector_storage`)
//! retrieves chunks by walking the knowledge graph: an entity is matched by
//! ANN and then its `source_chunk_ids` are pulled from vector storage. Text
//! chunks have entities pointing at them because the LLM saw their text;
//! figures don't, so they're unreachable through that path even though
//! they're indexed.
//!
//! Running each figure through the workspace's LLM with the same
//! entity-extraction prompt the text body uses produces entities whose
//! `source_chunk_ids` contain the figure's chunk id
//! (`{doc_id}-figure-{figure_id}`), and the existing entity-driven
//! retrieval surfaces the figure naturally.

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use edgequake_llm::traits::{ChatMessage, CompletionOptions, ImageData, LLMProvider};
use std::sync::Arc;

use crate::error::{PipelineError, Result};
use crate::extractor::ExtractionResult;
use crate::prompts::{default_entity_types, EntityExtractionPrompts, HybridExtractionParser};

/// Vision-capable entity extractor.
///
/// Generic over an [`LLMProvider`] so callers reuse whatever they already
/// constructed for text extraction (`safety_limits::create_safe_llm_provider`).
/// One instance per workspace.
pub struct VisionExtractionClient<L>
where
    L: LLMProvider + ?Sized,
{
    llm_provider: Arc<L>,
    entity_types: Vec<String>,
    prompts: EntityExtractionPrompts,
    parser: HybridExtractionParser,
    language: String,
    max_tokens: usize,
}

impl<L> VisionExtractionClient<L>
where
    L: LLMProvider + Send + Sync + ?Sized,
{
    /// Build a client from a workspace LLM provider. Defaults: standard
    /// entity types, English output, tuple-format prompts (matches
    /// `SOTAExtractor`), 4096 max output tokens.
    pub fn new(llm_provider: Arc<L>) -> Self {
        Self {
            llm_provider,
            entity_types: default_entity_types(),
            prompts: EntityExtractionPrompts::default(),
            parser: HybridExtractionParser::new(true),
            language: "English".to_string(),
            max_tokens: 4096,
        }
    }

    pub fn with_entity_types(mut self, types: Vec<String>) -> Self {
        self.entity_types = types;
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Model identifier of the underlying LLM provider — useful for logging.
    pub fn model(&self) -> &str {
        self.llm_provider.model()
    }

    /// Run entity extraction on a single figure.
    ///
    /// * `caption` — text from the figure's caption layout box. Becomes the
    ///   user-prompt text the LLM sees (same shape `SOTAExtractor` builds for
    ///   text chunks).
    /// * `image_bytes` / `mime` — figure PNG bytes (typically `image/png`).
    ///   Base64-encoded and sent as an `ImageData` content-part.
    /// * `chunk_id` — used as `source_chunk_id` on the returned result. Pass
    ///   `"{doc_id}-figure-{figure_id}"` so each entity's `source_chunk_ids`
    ///   gets linked to the figure-row id.
    pub async fn extract(
        &self,
        caption: &str,
        image_bytes: &[u8],
        mime: &str,
        chunk_id: &str,
    ) -> Result<ExtractionResult> {
        let start = std::time::Instant::now();

        let system_prompt = self
            .prompts
            .system_prompt(&self.entity_types, &self.language);
        // Reuse the same user prompt builder as the text path. Caption is
        // the "chunk content" the prompt frames its instructions around.
        let user_prompt =
            self.prompts
                .user_prompt(caption, &self.entity_types, &self.language);

        let image = ImageData::new(B64.encode(image_bytes), mime.to_string());

        let messages = vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user_with_images(user_prompt, vec![image]),
        ];

        // Deterministic extraction — same knobs SOTAExtractor uses. reasoning
        // models like gpt-5-mini default to allocating their entire budget to
        // CoT and producing empty content; reasoning_effort="none" disables
        // CoT for structured-output tasks. Non-reasoning models silently
        // ignore the field.
        let options = CompletionOptions {
            max_tokens: Some(self.max_tokens),
            temperature: Some(0.0),
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        };

        let response = self
            .llm_provider
            .chat(&messages, Some(&options))
            .await
            .map_err(|e| PipelineError::ExtractionError(format!("vision-extract chat: {e}")))?;

        let content = response.content.trim();
        if content.is_empty() {
            return Err(PipelineError::ExtractionError(format!(
                "vision-extract: empty content (finish_reason={:?}, completion_tokens={})",
                response.finish_reason, response.completion_tokens
            )));
        }

        let mut result = self.parser.parse(content, chunk_id)?;
        result.extraction_time_ms = start.elapsed().as_millis() as u64;
        result
            .metadata
            .insert("source".to_string(), serde_json::json!("vision_extract"));
        result.metadata.insert(
            "model".to_string(),
            serde_json::json!(self.llm_provider.model()),
        );
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgequake_llm::traits::{ChatRole, LLMResponse};
    use std::sync::Mutex;

    /// Mock LLM provider that records the last messages it saw and returns a
    /// fixed response. Lets us assert that the vision extractor builds the
    /// expected ChatMessage with images attached.
    #[derive(Default)]
    struct MockLLM {
        captured: Mutex<Option<Vec<ChatMessage>>>,
        response: String,
    }

    #[async_trait::async_trait]
    impl LLMProvider for MockLLM {
        fn name(&self) -> &str {
            "mock"
        }
        fn model(&self) -> &str {
            "mock-vision"
        }
        fn max_context_length(&self) -> usize {
            32_768
        }
        async fn complete(&self, _prompt: &str) -> edgequake_llm::Result<LLMResponse> {
            unimplemented!()
        }
        async fn complete_with_options(
            &self,
            _prompt: &str,
            _options: &CompletionOptions,
        ) -> edgequake_llm::Result<LLMResponse> {
            unimplemented!()
        }
        async fn chat(
            &self,
            messages: &[ChatMessage],
            _options: Option<&CompletionOptions>,
        ) -> edgequake_llm::Result<LLMResponse> {
            *self.captured.lock().unwrap() = Some(messages.to_vec());
            Ok(LLMResponse::new(self.response.clone(), "mock-vision"))
        }
    }

    #[tokio::test]
    async fn extract_passes_image_alongside_caption_via_user_with_images() {
        let mock = Arc::new(MockLLM {
            response: "entity<|#|>RED_PIXEL<|#|>concept<|#|>A single red pixel from the figure.\n<|COMPLETE|>".to_string(),
            ..Default::default()
        });
        let bytes = b"\x89PNG\r\n\x1a\n\x00fake-png-bytes".to_vec();
        let c = VisionExtractionClient::new(mock.clone());
        let result = c
            .extract(
                "Figure 1: A single red pixel.",
                &bytes,
                "image/png",
                "doc-figure-fig_0_0",
            )
            .await
            .expect("mock extract should succeed");

        // Captured messages: system + user_with_images.
        let captured = mock.captured.lock().unwrap().clone().expect("captured");
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].role, ChatRole::System);
        assert_eq!(captured[1].role, ChatRole::User);

        // The user message must carry exactly one image whose bytes round-trip
        // to the original input (proving the extractor isn't dropping image data).
        let images = captured[1]
            .images
            .as_ref()
            .expect("user message must carry images for the vision LLM");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].mime_type, "image/png");
        let decoded = B64.decode(&images[0].data).unwrap();
        assert_eq!(decoded, bytes);

        // Result: chunk_id propagates and the mock response was parsed.
        assert_eq!(result.source_chunk_id, "doc-figure-fig_0_0");
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].name, "RED_PIXEL");
    }

    /// Live integration test against the workspace's actual LLM via
    /// `edgequake_llm::ProviderFactory` — exercises the same resolution path
    /// production uses (`safety_limits::create_safe_llm_provider` wraps this
    /// factory call). Reads a real figure PNG + caption dumped from the dev
    /// chunks table.
    ///
    /// Prep:
    /// ```sh
    /// python3 -c "import psycopg2; ... write /tmp/fig-fig_4_0.bin and .caption"
    /// ```
    ///
    /// Run:
    /// ```sh
    /// EDGEQUAKE_VISION_TEST_PROVIDER=openai-compatible \
    /// EDGEQUAKE_VISION_TEST_MODEL='gemma 4 (E4B)' \
    /// OPENAI_COMPATIBLE_BASE_URL='http://localhost:8060/v1' \
    /// cargo test -p edgequake-pipeline --lib \
    ///   extractor::vision::tests::live_real_figure -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore = "requires live LLM + /tmp/fig-fig_4_0.bin"]
    async fn live_real_figure() {
        let png = std::fs::read("/tmp/fig-fig_4_0.bin")
            .expect("dump /tmp/fig-fig_4_0.bin before running this test");
        let caption = std::fs::read_to_string("/tmp/fig-fig_4_0.caption")
            .expect("dump /tmp/fig-fig_4_0.caption before running this test");

        let provider_name = std::env::var("EDGEQUAKE_VISION_TEST_PROVIDER")
            .unwrap_or_else(|_| "openai-compatible".to_string());
        let model = std::env::var("EDGEQUAKE_VISION_TEST_MODEL")
            .unwrap_or_else(|_| "gemma 4 (E4B)".to_string());
        let llm = edgequake_llm::factory::ProviderFactory::create_llm_provider(
            &provider_name,
            &model,
        )
        .expect("ProviderFactory::create_llm_provider");
        eprintln!(
            "LLM provider ready: name={} model={} (png={}B, caption={}chars)",
            llm.name(),
            llm.model(),
            png.len(),
            caption.len()
        );

        let client = VisionExtractionClient::new(llm).with_max_tokens(8192);
        let start = std::time::Instant::now();
        let result = client
            .extract(
                &caption,
                &png,
                "image/png",
                "4f7ce548-5021-4985-9602-51daa95cb31d-figure-fig_4_0",
            )
            .await
            .expect("vision-extract on real figure should succeed");

        eprintln!(
            "\n=== Vision extract on fig_4_0 ({:.1}s) ===",
            start.elapsed().as_secs_f32()
        );
        eprintln!("entities: {}", result.entities.len());
        for e in &result.entities {
            let desc = &e.description[..e.description.len().min(120)];
            eprintln!("  • {} [{}] — {}", e.name, e.entity_type, desc);
        }
        eprintln!("relationships: {}", result.relationships.len());
        for r in &result.relationships {
            let desc = &r.description[..r.description.len().min(80)];
            eprintln!(
                "  • {} -[{}]-> {}: {}",
                r.source, r.relation_type, r.target, desc
            );
        }

        assert_eq!(
            result.source_chunk_id,
            "4f7ce548-5021-4985-9602-51daa95cb31d-figure-fig_4_0"
        );
        assert!(
            !result.entities.is_empty(),
            "real figure must yield ≥1 entity — caption mentions GPT-2, TEE, ObfuscaTune"
        );
    }

    #[tokio::test]
    async fn empty_llm_response_surfaces_as_error() {
        let mock = Arc::new(MockLLM {
            response: "   ".to_string(),
            ..Default::default()
        });
        let c = VisionExtractionClient::new(mock);
        let err = c
            .extract("caption", b"png", "image/png", "doc-figure-fig_0_0")
            .await
            .expect_err("empty content must error");
        let msg = format!("{err}");
        assert!(msg.contains("empty content"), "got: {msg}");
    }
}
