//! Safety-limited LLM provider wrapper.
//!
//! This module provides a wrapper around any LLM provider that enforces
//! hard safety limits on token generation and request timeouts.
//!
//! Relocated from edgequake-llm to edgequake-api during the migration
//! to the external edgequake-llm crate (v0.2.1) which does not include
//! this application-level safety layer.

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

use edgequake_llm::{
    ChatMessage, CompletionOptions, ConfigProviderType, EmbeddingProvider, LLMProvider,
    LLMResponse, LlmError, ProviderConfig, ProviderFactory, Result,
};
use futures::stream::BoxStream;

/// Default maximum tokens for generation (8192).
pub const DEFAULT_MAX_TOKENS: usize = 8192;

/// Default request timeout in seconds (600 = 10 minutes).
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Absolute maximum tokens allowed (32768).
pub const ABSOLUTE_MAX_TOKENS: usize = 32768;

/// Minimum timeout in seconds (10).
pub const MINIMUM_TIMEOUT_SECS: u64 = 10;

/// Maximum timeout in seconds (600 = 10 minutes).
pub const MAXIMUM_TIMEOUT_SECS: u64 = 600;

/// Configuration for safety limits.
#[derive(Debug, Clone)]
pub struct SafetyLimitsConfig {
    /// Maximum tokens to generate per request.
    pub max_tokens: usize,
    /// Request timeout.
    pub timeout: Duration,
    /// Whether to log when limits are enforced.
    pub log_enforcement: bool,
}

impl Default for SafetyLimitsConfig {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            log_enforcement: true,
        }
    }
}

impl SafetyLimitsConfig {
    /// Create a new config with custom limits.
    pub fn new(max_tokens: usize, timeout_secs: u64) -> Self {
        Self {
            max_tokens: max_tokens.clamp(1, ABSOLUTE_MAX_TOKENS),
            timeout: Duration::from_secs(
                timeout_secs.clamp(MINIMUM_TIMEOUT_SECS, MAXIMUM_TIMEOUT_SECS),
            ),
            log_enforcement: true,
        }
    }

    /// Create config from environment variables.
    pub fn from_env() -> Self {
        let max_tokens = std::env::var("EDGEQUAKE_LLM_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_TOKENS)
            .clamp(1, ABSOLUTE_MAX_TOKENS);

        let timeout_secs = std::env::var("EDGEQUAKE_LLM_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .clamp(MINIMUM_TIMEOUT_SECS, MAXIMUM_TIMEOUT_SECS);

        Self {
            max_tokens,
            timeout: Duration::from_secs(timeout_secs),
            log_enforcement: true,
        }
    }

    /// Create a strict config for testing (low limits).
    pub fn strict() -> Self {
        Self {
            max_tokens: 1024,
            timeout: Duration::from_secs(30),
            log_enforcement: true,
        }
    }

    /// Create a permissive config (high limits).
    pub fn permissive() -> Self {
        Self {
            max_tokens: ABSOLUTE_MAX_TOKENS,
            timeout: Duration::from_secs(MAXIMUM_TIMEOUT_SECS),
            log_enforcement: true,
        }
    }

    /// Disable enforcement logging.
    pub fn without_logging(mut self) -> Self {
        self.log_enforcement = false;
        self
    }
}

/// Safety-limited LLM provider wrapper that works with `Arc<dyn LLMProvider>`.
pub struct SafetyLimitedProviderWrapper {
    inner: Arc<dyn LLMProvider>,
    config: SafetyLimitsConfig,
}

impl SafetyLimitedProviderWrapper {
    /// Create a new safety-limited provider wrapper.
    pub fn new(provider: Arc<dyn LLMProvider>, config: SafetyLimitsConfig) -> Self {
        Self {
            inner: provider,
            config,
        }
    }

    /// Apply max_tokens limit to options.
    fn apply_token_limit(&self, options: &CompletionOptions) -> CompletionOptions {
        let mut opts = options.clone();

        let requested = opts.max_tokens.unwrap_or(self.config.max_tokens);
        let effective = requested.min(self.config.max_tokens);

        if requested != effective && self.config.log_enforcement {
            tracing::warn!(
                requested_tokens = requested,
                enforced_tokens = effective,
                "Safety limit: max_tokens clamped to configured limit"
            );
        }

        opts.max_tokens = Some(effective);
        opts
    }
}

#[async_trait]
impl LLMProvider for SafetyLimitedProviderWrapper {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn max_context_length(&self) -> usize {
        self.inner.max_context_length()
    }

    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let options = CompletionOptions {
            max_tokens: Some(self.config.max_tokens),
            ..Default::default()
        };

        self.complete_with_options(prompt, &options).await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<LLMResponse> {
        let safe_options = self.apply_token_limit(options);

        let result = tokio::time::timeout(
            self.config.timeout,
            self.inner.complete_with_options(prompt, &safe_options),
        )
        .await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_elapsed) => {
                if self.config.log_enforcement {
                    tracing::error!(
                        timeout_secs = self.config.timeout.as_secs(),
                        "Safety limit: LLM request timed out"
                    );
                }
                Err(LlmError::Timeout)
            }
        }
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        options: Option<&CompletionOptions>,
    ) -> Result<LLMResponse> {
        let default_options = CompletionOptions {
            max_tokens: Some(self.config.max_tokens),
            ..Default::default()
        };

        let safe_options = match options {
            Some(opts) => self.apply_token_limit(opts),
            None => default_options,
        };

        let result = tokio::time::timeout(
            self.config.timeout,
            self.inner.chat(messages, Some(&safe_options)),
        )
        .await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_elapsed) => {
                if self.config.log_enforcement {
                    tracing::error!(
                        timeout_secs = self.config.timeout.as_secs(),
                        message_count = messages.len(),
                        "Safety limit: LLM chat request timed out"
                    );
                }
                Err(LlmError::Timeout)
            }
        }
    }

    async fn stream(&self, prompt: &str) -> Result<BoxStream<'static, Result<String>>> {
        let result = tokio::time::timeout(self.config.timeout, self.inner.stream(prompt)).await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_elapsed) => {
                if self.config.log_enforcement {
                    tracing::error!(
                        timeout_secs = self.config.timeout.as_secs(),
                        "Safety limit: LLM stream request timed out"
                    );
                }
                Err(LlmError::Timeout)
            }
        }
    }

    fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }
}

/// Safety-limited embedding provider wrapper that works with `Arc<dyn EmbeddingProvider>`.
pub struct SafetyLimitedEmbeddingProviderWrapper {
    inner: Arc<dyn EmbeddingProvider>,
    config: SafetyLimitsConfig,
}

impl SafetyLimitedEmbeddingProviderWrapper {
    /// Create a new safety-limited embedding provider wrapper.
    pub fn new(provider: Arc<dyn EmbeddingProvider>, config: SafetyLimitsConfig) -> Self {
        Self {
            inner: provider,
            config,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for SafetyLimitedEmbeddingProviderWrapper {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn max_tokens(&self) -> usize {
        self.inner.max_tokens()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let result = tokio::time::timeout(self.config.timeout, self.inner.embed(texts)).await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_elapsed) => {
                if self.config.log_enforcement {
                    tracing::error!(
                        timeout_secs = self.config.timeout.as_secs(),
                        text_count = texts.len(),
                        "Safety limit: Embedding request timed out"
                    );
                }
                Err(LlmError::Timeout)
            }
        }
    }
}

/// Validate that the required API key environment variable is set and non-empty for the
/// given provider, returning a clear `ConfigError` before attempting to build the client.
fn check_api_key(provider_name: &str) -> Result<()> {
    let (env_var, display_name) = match provider_name {
        "openai" => ("OPENAI_API_KEY", "OpenAI"),
        "anthropic" => ("ANTHROPIC_API_KEY", "Anthropic"),
        "gemini" => ("GEMINI_API_KEY", "Gemini"),
        "mistral" => ("MISTRAL_API_KEY", "Mistral"),
        "xai" => ("XAI_API_KEY", "xAI"),
        "openrouter" => ("OPENROUTER_API_KEY", "OpenRouter"),
        _ => return Ok(()), // Local / key-less providers (ollama, lmstudio, mock, etc.)
    };
    let key_present = std::env::var(env_var)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    if !key_present {
        return Err(LlmError::ConfigError(format!(
            "{env_var} is not set. To use the {display_name} provider, \
             set the environment variable and restart the server. \
             Alternatively, select the Ollama provider which runs locally."
        )));
    }
    Ok(())
}

/// Create an LLM provider, using EDGEQUAKE_LLM_TIMEOUT to override the 120s default
/// when the openai-compatible provider is used with local models.
fn create_llm_provider_with_timeout(
    provider_name: &str,
    model: &str,
) -> Result<Arc<dyn LLMProvider>> {
    let timeout: Option<u64> = std::env::var("EDGEQUAKE_LLM_TIMEOUT")
        .ok()
        .and_then(|v| v.parse().ok());

    // Only override for openai-compatible when timeout is explicitly set
    if let Some(timeout_secs) = timeout {
        if provider_name == "openai-compatible" {
            if let Ok(base_url) = std::env::var("OPENAI_COMPATIBLE_BASE_URL") {
                tracing::info!(
                    provider = provider_name,
                    model = model,
                    timeout_secs = timeout_secs,
                    base_url = %base_url,
                    "Using custom LLM timeout for openai-compatible provider"
                );
                let mut config = ProviderConfig {
                    name: "openai-compatible".to_string(),
                    display_name: "OpenAI Compatible".to_string(),
                    provider_type: ConfigProviderType::OpenAICompatible,
                    base_url: Some(base_url),
                    default_llm_model: Some(model.to_string()),
                    timeout_seconds: timeout_secs,
                    ..Default::default()
                };
                if let Ok(api_key) = std::env::var("OPENAI_COMPATIBLE_API_KEY") {
                    if !api_key.is_empty() {
                        config.api_key = Some(api_key);
                    }
                }
                let (llm, _) = ProviderFactory::from_config_with_model(&config, Some(model))?;
                return Ok(llm);
            }
        }
    }

    // Default path for all other cases
    ProviderFactory::create_llm_provider(provider_name, model)
}

/// Create a safety-limited LLM provider from workspace configuration.
pub fn create_safe_llm_provider(provider_name: &str, model: &str) -> Result<Arc<dyn LLMProvider>> {
    check_api_key(provider_name)?;
    let inner = create_llm_provider_with_timeout(provider_name, model)?;
    let config = SafetyLimitsConfig::from_env();

    tracing::info!(
        provider = provider_name,
        model = model,
        max_tokens = config.max_tokens,
        timeout_secs = config.timeout.as_secs(),
        "Creating safety-limited LLM provider"
    );

    Ok(Arc::new(SafetyLimitedProviderWrapper::new(inner, config)))
}

/// Create a safety-limited embedding provider from workspace configuration.
pub fn create_safe_embedding_provider(
    provider_name: &str,
    model: &str,
    dimension: usize,
) -> Result<Arc<dyn EmbeddingProvider>> {
    let inner = ProviderFactory::create_embedding_provider(provider_name, model, dimension)?;
    let config = SafetyLimitsConfig::from_env();

    tracing::info!(
        provider = provider_name,
        model = model,
        dimension = dimension,
        timeout_secs = config.timeout.as_secs(),
        "Creating safety-limited embedding provider"
    );

    Ok(Arc::new(SafetyLimitedEmbeddingProviderWrapper::new(
        inner, config,
    )))
}

// ─────────────────────────────────────────────────────────────────────────────
// Vision / PDF provider helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum allowed outer PDF-vision conversion timeout (24 hours).
///
/// This is a sanity upper-bound only. Vision extraction for very large documents
/// (1 000+ pages) with local models can legitimately take hours.
pub const VISION_MAX_OUTER_TIMEOUT_SECS: u64 = 86_400;

/// Returns `true` when `provider_name` refers to a local, in-process inference
/// server (Ollama, LM Studio, …) rather than a cloud API.
///
/// Local providers are memory-bound rather than network-bound, so they need
/// longer per-page timeouts and lower concurrency.
pub fn is_local_provider(provider_name: &str) -> bool {
    matches!(
        provider_name.to_ascii_lowercase().as_str(),
        "ollama" | "lmstudio" | "lm-studio" | "lm_studio" | "mock"
    )
}

/// Returns the recommended default seconds-per-page for the given provider.
///
/// Reads `EDGEQUAKE_PDF_SECS_PER_PAGE` first; falls back to:
/// - Local providers: 30 s / page (conservative for a mid-range GPU)
/// - Cloud providers:  8 s / page
pub fn secs_per_page_for_provider(provider_name: &str) -> u64 {
    if let Ok(val) = std::env::var("EDGEQUAKE_PDF_SECS_PER_PAGE") {
        if let Ok(n) = val.parse::<u64>() {
            // Enforce a floor of 5 s to prevent accidentally tiny timeouts.
            return n.max(5);
        }
    }
    if is_local_provider(provider_name) {
        30
    } else {
        8
    }
}

/// Compute the outer vision-conversion timeout for the entire PDF.
///
/// Formula: `120 + (page_count × secs_per_page_for_provider(provider))`
/// clamped to `VISION_MAX_OUTER_TIMEOUT_SECS`.
pub fn vision_outer_timeout_secs(provider_name: &str, page_count: usize) -> u64 {
    let per_page = secs_per_page_for_provider(provider_name);
    let computed = 120_u64.saturating_add(per_page.saturating_mul(page_count as u64));
    computed.min(VISION_MAX_OUTER_TIMEOUT_SECS)
}

/// Returns the per-page LLM call timeout for vision/OCR requests.
///
/// Reads `EDGEQUAKE_VISION_PAGE_TIMEOUT_SECS` first; falls back to:
/// - Local providers: 600 s per page (no hard upper cap applied here)
/// - Cloud providers: 120 s per page
///
/// Unlike `create_safe_llm_provider`, this value is NOT clamped to
/// `MAXIMUM_TIMEOUT_SECS` so that local providers can handle slow pages.
pub fn vision_page_timeout_secs(provider_name: &str) -> u64 {
    if let Ok(val) = std::env::var("EDGEQUAKE_VISION_PAGE_TIMEOUT_SECS") {
        if let Ok(n) = val.parse::<u64>() {
            return n.max(10);
        }
    }
    if is_local_provider(provider_name) {
        600
    } else {
        120
    }
}

/// Create a safety-limited LLM provider suitable for **vision/PDF OCR** calls.
///
/// Unlike [`create_safe_llm_provider`] (which caps timeouts at `MAXIMUM_TIMEOUT_SECS`),
/// this function derives the per-page timeout from [`vision_page_timeout_secs`] so that
/// local providers (Ollama, LM Studio) are not artificially cut off mid-page.
///
/// # Usage
/// ```ignore
/// let provider = create_safe_vision_provider("ollama", "glm-ocr:latest")?;
/// ```
pub fn create_safe_vision_provider(
    provider_name: &str,
    model: &str,
) -> Result<Arc<dyn LLMProvider>> {
    check_api_key(provider_name)?;
    let inner = ProviderFactory::create_llm_provider(provider_name, model)?;

    let timeout_secs = vision_page_timeout_secs(provider_name);
    let config = SafetyLimitsConfig {
        max_tokens: DEFAULT_MAX_TOKENS,
        timeout: Duration::from_secs(timeout_secs),
        log_enforcement: true,
    };

    tracing::info!(
        provider = provider_name,
        model = model,
        timeout_secs = timeout_secs,
        is_local = is_local_provider(provider_name),
        "Creating safety-limited VISION LLM provider (provider-aware timeout)"
    );

    Ok(Arc::new(SafetyLimitedProviderWrapper::new(inner, config)))
}

// ─────────────────────────────────────────────────────────────────────────────

/// Get the default model for a given provider name.
pub fn default_model_for_provider(provider_name: &str) -> &'static str {
    match provider_name.to_lowercase().as_str() {
        "openai" => "gpt-4.1-nano",
        "anthropic" => "claude-sonnet-4-5-20250929",
        "gemini" => "gemini-2.5-flash",
        "xai" => "grok-4-1-fast",
        "openrouter" => "openai/gpt-4o-mini",
        "ollama" => "gemma3:12b",
        "lmstudio" | "lm-studio" | "lm_studio" => "gemma-3n-e4b-it",
        "minimax" => "MiniMax-M2.7",
        "mock" => "mock-model",
        _ => "gpt-4.1-nano",
    }
}
