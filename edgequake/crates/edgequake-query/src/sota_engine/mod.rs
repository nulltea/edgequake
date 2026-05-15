//! SOTA Query Engine - LightRAG-inspired implementation.
//!
//! # Implements
//!
//! - **FEAT0007**: Multi-Mode Query Execution
//! - **FEAT0101**: Naive Mode (vector search only)
//! - **FEAT0102**: Local Mode (entity-centric)
//! - **FEAT0103**: Global Mode (community summaries)
//! - **FEAT0104**: Hybrid Mode (local + global)
//! - **FEAT0105**: Mix Mode (adaptive blend)
//! - **FEAT0106**: Bypass Mode (direct LLM)
//! - **FEAT0107**: LLM-Based Keyword Extraction
//! - **FEAT0108**: Smart Context Truncation
//! - **FEAT0109**: SOTA Query Delegation
//!
//! # Enforces
//!
//! - **BR0101**: Token budget must not exceed LLM context window
//! - **BR0102**: Graph context takes priority over naive chunks
//! - **BR0103**: Query mode must be valid enum value
//! - **BR0104**: Conversation history included in context
//! - **BR0106**: Keyword cache TTL 24 hours default
//!
//! This module provides the enhanced query engine with:
//! - LLM-based keyword extraction with caching
//! - Mode-specific vector search (entities vs relationships)
//! - Batch graph operations
//! - Query caching
//!
//! # Architecture
//!
//! ```text
//! Query → Keyword Extraction → Mode Router
//!                                 ↓
//!         ┌───────────────────────┼───────────────────────┐
//!         ↓                       ↓                       ↓
//!     Local Mode             Global Mode             Naive Mode
//!   (Entity VDB +          (Relationship VDB +      (Chunk VDB)
//!    low-level kw)          high-level kw)
//!         ↓                       ↓                       ↓
//!         └───────────────────────┼───────────────────────┘
//!                                 ↓
//!                         Context Building
//!                                 ↓
//!                         Token Budgeting
//!                                 ↓
//!                         LLM Generation
//! ```
//!
//! # WHY: LightRAG Algorithm
//!
//! This implements the LightRAG paper's multi-level retrieval strategy:
//!
//! 1. **Keyword Extraction**: LLM extracts high-level (themes) and low-level
//!    (entities) keywords from the query. WHY: Different keywords retrieve
//!    different context types optimally.
//!
//! 2. **Mode-Specific Search**:
//!    - Local: Uses low-level keywords to find entity nodes
//!    - Global: Uses high-level keywords to find relationship clusters
//!    - Naive: Direct query embedding against chunk vectors
//!
//! 3. **Token Budgeting**: Context is truncated to fit LLM window while
//!    maintaining the most relevant information. Graph context is prioritized
//!    over raw chunks because graph relationships are pre-summarized.
//!
//! # See Also
//!
//! - [`QueryMode`] for available modes
//! - [`QueryRequest`] for query parameters
//! - [docs/features.md](../../../../../../docs/features.md) for feature details

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{QueryError, Result};
use crate::keywords::{
    CachedKeywordExtractor, ExtractedKeywords, InMemoryKeywordCache, KeywordExtractor,
    LLMKeywordExtractor, MockKeywordExtractor,
};
use crate::modes::QueryMode;
use crate::tokenizer::{SimpleTokenizer, Tokenizer};
use crate::truncation::TruncationConfig;

use edgequake_agents::code_analysis::JinaEmbedder;
use edgequake_llm::traits::{EmbeddingProvider, LLMProvider};
use edgequake_llm::Reranker;
use edgequake_storage::traits::{
    AlgorithmVectorStorage, CodeVectorStorage, GraphStorage, VectorStorage,
};

/// Configuration for the SOTA query engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOTAQueryConfig {
    /// Default query mode.
    pub default_mode: QueryMode,

    /// Maximum entities to retrieve.
    pub max_entities: usize,

    /// Maximum relationships to retrieve.
    pub max_relationships: usize,

    /// Maximum chunks to retrieve.
    pub max_chunks: usize,

    /// Maximum context tokens.
    pub max_context_tokens: usize,

    /// Graph traversal depth.
    pub graph_depth: usize,

    /// Minimum cosine similarity for entity / relationship vector matches.
    /// Used by the graph-side retrieval paths (Local entity-vector,
    /// Global relationship-vector). Calibrated for entity-vector noise.
    pub min_score: f32,

    /// Minimum cosine similarity for chunk vector matches.
    ///
    /// Chunk embeddings sit at a higher baseline cosine than entity/relationship
    /// embeddings against typical technical corpora — for embedders like
    /// qwen3-embedding:0.6b, irrelevant queries still produce ~0.30 cosine on
    /// chunks. A separate floor keeps the chunk recall tight without
    /// over-filtering graph nodes.
    pub chunk_min_score: f32,

    /// Whether to use keyword extraction.
    pub use_keyword_extraction: bool,

    /// Whether to use adaptive mode selection based on query intent.
    pub use_adaptive_mode: bool,

    /// Truncation configuration.
    pub truncation: TruncationConfig,

    /// Keyword cache TTL in seconds.
    pub keyword_cache_ttl_secs: u64,

    /// Enable the in-memory BM25 reranker on the retrieved chunk set.
    /// BM25 rescores candidates against the query, boosting chunks that
    /// contain rare query tokens (e.g. proper nouns) that pure vector
    /// cosine smears into a semantic cluster. Workspaces can override via
    /// `workspace.enable_rerank` → `QueryRequest::enable_rerank`.
    pub enable_rerank: bool,

    /// Number of top candidates returned after reranking (also the upper
    /// bound on candidates considered).
    pub rerank_top_k: usize,

    /// Minimum BM25 rerank score required to keep a chunk. If every
    /// candidate falls below this floor, the rerank step falls back to
    /// the original (unsorted) top-K to preserve recall.
    pub min_rerank_score: f32,

    /// Engine default task description for the Qwen3-Embedding query
    /// instruction prefix. Queries (and the LightRAG high-/low-level
    /// keyword strings) are wrapped as `Instruct: {task}\nQuery: {q}`
    /// before embedding. Workspaces can override via
    /// `QueryRequest::query_instruction`; an empty string disables the
    /// wrapper. Document embeddings are NOT prefixed (Qwen3-Embedding
    /// is trained asymmetric).
    #[serde(default = "crate::engine::default_query_instruction")]
    pub default_query_instruction: String,
}

impl Default for SOTAQueryConfig {
    fn default() -> Self {
        Self {
            default_mode: QueryMode::Hybrid,
            // WHY 60: LightRAG uses top_k=60 entities. More entity candidates = more
            // chunk candidates from the KG path, directly improving recall.
            max_entities: 60,
            // WHY 60: Match entity count for balanced KG context.
            // LightRAG allocates max_relation_tokens=8000 for relations.
            max_relationships: 60,
            // WHY 20: LightRAG uses chunk_top_k=20. More text chunks = more direct
            // evidence for the LLM, improving both recall and correctness.
            max_chunks: 20,
            // WHY 30000: LightRAG uses max_total_tokens=30000. With gpt-4o-mini
            // having 128K context, 4000 tokens was throwing away ~87% of usable context.
            // 30000 tokens uses only 23% of the context window — safe and effective.
            max_context_tokens: 30000,
            graph_depth: 2,
            min_score: 0.1,
            // WHY 0.4: Calibrated against qwen3-embedding:0.6b on a technical corpus.
            // Empirically, irrelevant queries plateau around 0.28-0.32 chunk cosine
            // while relevant chunks score 0.46+. A 0.4 floor cleanly separates the
            // two regimes. Retune per workspace/embedder if false negatives appear.
            chunk_min_score: 0.4,
            use_keyword_extraction: true,
            use_adaptive_mode: true,
            // WHY derived from max_context_tokens: The truncation budget MUST match
            // the context token budget, otherwise the system fetches chunks it then
            // throws away. LightRAG splits: 50% entities, 50% relationships, chunks
            // fill the remainder. With 30K total: entities=10K, rels=10K, chunks=10K.
            truncation: TruncationConfig {
                max_entity_tokens: 10000,
                max_relation_tokens: 10000,
                max_total_tokens: 30000,
            },
            keyword_cache_ttl_secs: 24 * 60 * 60, // 24 hours
            enable_rerank: true,
            rerank_top_k: 20,
            // WHY 0.1: BM25 scores can be low for short documents or simple
            // queries; 0.3 was too aggressive and filtered out valid chunks.
            // 0.1 separates clear non-matches from anything with some keyword
            // overlap. Combined with the rerank-empty fallback below.
            min_rerank_score: 0.1,
            default_query_instruction: crate::engine::default_query_instruction(),
        }
    }
}

/// Query embeddings for different keyword levels.
///
/// LightRAG uses different embeddings for different modes:
/// - low_level: Entity search (Local mode)
/// - high_level: Relationship search (Global mode)
/// - query: Direct chunk search (Naive mode)
pub struct QueryEmbeddings {
    /// Original query embedding.
    pub query: Vec<f32>,

    /// High-level keywords embedding (for Global mode).
    pub high_level: Vec<f32>,

    /// Low-level keywords embedding (for Local mode).
    pub low_level: Vec<f32>,
}

impl QueryEmbeddings {
    /// Compute all embeddings in a single batch.
    ///
    /// `task` is the Qwen3-Embedding instruction task description (see
    /// `crate::engine::wrap_query_instruction`). All three query-side
    /// texts get the prefix so they live in the model's query subspace;
    /// document embeddings stay raw at indexing time.
    pub async fn compute(
        query: &str,
        keywords: &ExtractedKeywords,
        embedder: &dyn EmbeddingProvider,
        task: &str,
    ) -> Result<Self> {
        let high_level_text = if keywords.high_level.is_empty() {
            query.to_string()
        } else {
            keywords.high_level.join(", ")
        };

        let low_level_text = if keywords.low_level.is_empty() {
            query.to_string()
        } else {
            keywords.low_level.join(", ")
        };

        // Batch embed all three texts with the instruction prefix applied
        // uniformly (research recommendation: start unified across chunk,
        // entity, and relationship retrieval paths).
        let texts = vec![
            crate::engine::wrap_query_instruction(query, task),
            crate::engine::wrap_query_instruction(&high_level_text, task),
            crate::engine::wrap_query_instruction(&low_level_text, task),
        ];

        let embeddings = embedder.embed(&texts).await.map_err(QueryError::from)?;

        if embeddings.len() != 3 {
            return Err(QueryError::Internal(format!(
                "Expected 3 embeddings, got {}",
                embeddings.len()
            )));
        }

        Ok(Self {
            query: embeddings[0].clone(),
            high_level: embeddings[1].clone(),
            low_level: embeddings[2].clone(),
        })
    }

    /// Simple embedding (same for all levels).
    pub fn uniform(embedding: Vec<f32>) -> Self {
        Self {
            query: embedding.clone(),
            high_level: embedding.clone(),
            low_level: embedding,
        }
    }
}

/// Configuration for the optional semantic (HTTP cross-encoder) reranker.
///
/// One `SemanticRerankerConfig` is built at startup from `RERANKER_URL` /
/// `RERANKER_MODEL` / etc. At query time, `SOTAQueryEngine` consults a
/// per-model lazy cache of `HttpReranker` instances so workspaces can pick
/// a different `model` name (e.g. `"bge-reranker-v2-m3"` vs the env default
/// `"jina-reranker-v3"`) without paying a reqwest-client setup cost on
/// every request.
#[derive(Debug, Clone)]
pub struct SemanticRerankerConfig {
    /// Default model name when the per-request override is `None`.
    pub default_model: String,
    /// Full `/v1/rerank` endpoint URL (e.g. llama-swap front).
    pub base_url: String,
    /// Optional bearer token for the rerank API.
    pub api_key: Option<String>,
    /// Per-request HTTP timeout.
    pub timeout: std::time::Duration,
}

pub struct SOTAQueryEngine {
    config: SOTAQueryConfig,
    vector_storage: Arc<dyn VectorStorage>,
    graph_storage: Arc<dyn GraphStorage>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    llm_provider: Arc<dyn LLMProvider>,
    keyword_extractor: Arc<dyn KeywordExtractor>,
    tokenizer: Arc<dyn Tokenizer>,
    /// Cache for keyword validation (keyword -> exists_in_graph).
    /// WHY: Avoids repeated graph lookups for the same keywords.
    keyword_validation_cache: Arc<tokio::sync::RwLock<std::collections::HashMap<String, bool>>>,
    /// Phase 1 Reference Code GraphRAG: approved-snippet vector store.
    /// `None` disables the post-retrieval code-enrichment step entirely.
    code_vector_storage: Option<Arc<dyn CodeVectorStorage>>,
    /// Jina code-embedder client used to embed the NL query for the same step.
    /// `None` disables enrichment just like an absent `code_vector_storage`.
    code_embedder: Option<Arc<JinaEmbedder>>,
    /// Approved-algorithm vector store used by the sibling enrichment pass.
    /// `None` disables the post-retrieval algorithm-enrichment step.
    algorithm_vector_storage: Option<Arc<dyn AlgorithmVectorStorage>>,
    /// Default reranker (BM25 in production). `None` skips the rerank step.
    /// Wired via [`Self::with_reranker`].
    reranker: Option<Arc<dyn Reranker>>,

    /// Optional cross-encoder HTTP reranker config, selected when the
    /// per-request `reranker_strategy == "semantic"`. `None` falls back to
    /// `reranker`. Wired via [`Self::with_semantic_config`].
    semantic_config: Option<SemanticRerankerConfig>,
    /// Per-model cache of `HttpReranker` instances keyed by model name.
    /// Lazily populated on first request for each model; entries share the
    /// engine's `semantic_config` for URL / api_key / timeout.
    semantic_cache: tokio::sync::RwLock<std::collections::HashMap<String, Arc<dyn Reranker>>>,
}

impl SOTAQueryEngine {
    /// Create a new SOTA query engine.
    pub fn new(
        config: SOTAQueryConfig,
        vector_storage: Arc<dyn VectorStorage>,
        graph_storage: Arc<dyn GraphStorage>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        llm_provider: Arc<dyn LLMProvider>,
    ) -> Self {
        // Create cached keyword extractor
        let base_extractor = Arc::new(LLMKeywordExtractor::new(llm_provider.clone()));
        let cache = Arc::new(InMemoryKeywordCache::new(1000));
        let keyword_extractor: Arc<dyn KeywordExtractor> = Arc::new(CachedKeywordExtractor::new(
            base_extractor,
            cache,
            std::time::Duration::from_secs(config.keyword_cache_ttl_secs),
        ));

        Self {
            config,
            vector_storage,
            graph_storage,
            embedding_provider,
            llm_provider,
            keyword_extractor,
            tokenizer: Arc::new(SimpleTokenizer),
            keyword_validation_cache: Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            code_vector_storage: None,
            code_embedder: None,
            algorithm_vector_storage: None,
            reranker: None,
            semantic_config: None,
            semantic_cache: tokio::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Create with mock keyword extractor (for testing).
    pub fn with_mock_keywords(
        config: SOTAQueryConfig,
        vector_storage: Arc<dyn VectorStorage>,
        graph_storage: Arc<dyn GraphStorage>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        llm_provider: Arc<dyn LLMProvider>,
    ) -> Self {
        let keyword_extractor: Arc<dyn KeywordExtractor> = Arc::new(MockKeywordExtractor::new());

        Self {
            config,
            vector_storage,
            graph_storage,
            embedding_provider,
            llm_provider,
            keyword_extractor,
            tokenizer: Arc::new(SimpleTokenizer),
            keyword_validation_cache: Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            code_vector_storage: None,
            code_embedder: None,
            algorithm_vector_storage: None,
            reranker: None,
            semantic_config: None,
            semantic_cache: tokio::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Set a custom keyword extractor.
    pub fn with_keyword_extractor(mut self, extractor: Arc<dyn KeywordExtractor>) -> Self {
        self.keyword_extractor = extractor;
        self
    }

    /// Set a custom tokenizer.
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Wire in the approved-code vector store + code embedder used by the
    /// Phase 1 Reference Code GraphRAG post-retrieval enrichment step.
    pub fn with_code_reference(
        mut self,
        storage: Arc<dyn CodeVectorStorage>,
        embedder: Arc<JinaEmbedder>,
    ) -> Self {
        self.code_vector_storage = Some(storage);
        self.code_embedder = Some(embedder);
        self
    }

    /// Accessor for the code-reference vector store (None = feature off).
    pub fn code_vector_storage(&self) -> Option<&Arc<dyn CodeVectorStorage>> {
        self.code_vector_storage.as_ref()
    }

    /// Accessor for the code embedder (None = feature off).
    pub fn code_embedder(&self) -> Option<&Arc<JinaEmbedder>> {
        self.code_embedder.as_ref()
    }

    /// Wire in the approved-algorithm vector store used by the
    /// post-retrieval algorithm enrichment step (symmetric with
    /// [`Self::with_code_reference`]).
    pub fn with_approved_algorithms(mut self, storage: Arc<dyn AlgorithmVectorStorage>) -> Self {
        self.algorithm_vector_storage = Some(storage);
        self
    }

    /// Accessor for the algorithm vector store (None = feature off).
    pub fn algorithm_vector_storage(&self) -> Option<&Arc<dyn AlgorithmVectorStorage>> {
        self.algorithm_vector_storage.as_ref()
    }

    /// Wire in the BM25 reranker that rescores retrieved chunks against
    /// the query keyword set. See [`Self::rerank_chunks_with_strategy`].
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Wire in the cross-encoder HTTP reranker config that is selected when
    /// a query's `reranker_strategy == "semantic"`. Coexists with the
    /// default (BM25) reranker — workspace config picks per-query, and the
    /// engine lazily caches one `HttpReranker` instance per `model` name so
    /// workspaces can override `reranker_model`.
    pub fn with_semantic_config(mut self, config: SemanticRerankerConfig) -> Self {
        self.semantic_config = Some(config);
        self
    }

    /// Resolve (and lazily create + cache) the `HttpReranker` for a given
    /// model name. Returns `None` when no semantic config is wired.
    ///
    /// `pub` so integration tests in `tests/` can exercise the registry
    /// directly — production callers should go through `rerank_chunks_with_strategy`.
    #[doc(hidden)]
    pub async fn semantic_reranker_for_model(
        &self,
        model: Option<&str>,
    ) -> Option<Arc<dyn Reranker>> {
        let cfg = self.semantic_config.as_ref()?;
        let model = model.unwrap_or(&cfg.default_model).to_string();

        // Fast path: read lock, hit
        {
            let cache = self.semantic_cache.read().await;
            if let Some(r) = cache.get(&model) {
                return Some(Arc::clone(r));
            }
        }

        // Slow path: write lock, re-check (someone may have inserted), then create
        let mut cache = self.semantic_cache.write().await;
        if let Some(r) = cache.get(&model) {
            return Some(Arc::clone(r));
        }

        // Cloud rerankers (Jina, Cohere, Aliyun) already return [0,1]
        // relevance scores. Anything else (llama.cpp / llama-swap fronts,
        // self-hosted endpoints) is a cross-encoder emitting raw classifier
        // logits — apply sigmoid so the engine's `min_rerank_score` floor
        // (tuned for [0,1]) stays meaningful regardless of which reranker
        // the workspace selects.
        //
        // Detection is on `base_url`, not model name: e.g. `jina-reranker-v3`
        // exists both as a cloud API (api.jina.ai → normalized) and as a
        // local llama.cpp build (llama-swap → raw logits). Same model name,
        // different score scale.
        let host = cfg.base_url.to_ascii_lowercase();
        let cloud_normalized = host.contains("api.jina.ai")
            || host.contains("api.cohere.com")
            || host.contains("dashscope.aliyuncs.com");
        let rerank_config = edgequake_llm::reranker::RerankConfig {
            model: model.clone(),
            base_url: cfg.base_url.clone(),
            api_key: cfg.api_key.clone(),
            top_n: None,
            timeout: cfg.timeout,
            enable_chunking: false,
            max_tokens_per_doc: 480,
            sigmoid_normalize: !cloud_normalized,
        };
        let new: Arc<dyn Reranker> =
            Arc::new(edgequake_llm::reranker::HttpReranker::new(rerank_config));
        cache.insert(model, Arc::clone(&new));
        Some(new)
    }
}

impl SOTAQueryEngine {
    /// Get the query configuration.
    pub fn config(&self) -> &SOTAQueryConfig {
        &self.config
    }

    /// Resolve the Qwen3-Embedding instruction task description for a
    /// request: the per-request override if set, else the engine default.
    pub(crate) fn resolved_query_instruction<'a>(
        &'a self,
        request: &'a crate::engine::QueryRequest,
    ) -> &'a str {
        request
            .query_instruction
            .as_deref()
            .unwrap_or(&self.config.default_query_instruction)
    }
}

mod keyword_validation;
mod prompt;
mod query_entry;
mod query_modes;
mod reranking;
mod vector_queries;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sota_config_default() {
        let config = SOTAQueryConfig::default();
        assert_eq!(config.default_mode, QueryMode::Hybrid);
        assert!(config.use_keyword_extraction);
        assert!(config.use_adaptive_mode);
    }

    #[test]
    fn test_query_embeddings_uniform() {
        let embedding = vec![1.0, 2.0, 3.0];
        let embeddings = QueryEmbeddings::uniform(embedding.clone());

        assert_eq!(embeddings.query, embedding);
        assert_eq!(embeddings.high_level, embedding);
        assert_eq!(embeddings.low_level, embedding);
    }

    /// @implements SPEC-004: build_prompt with system_prompt_extension
    mod system_prompt_tests {
        use super::*;
        use crate::context::{QueryContext, RetrievedChunk};
        use edgequake_llm::MockProvider;
        use edgequake_storage::{MemoryGraphStorage, MemoryVectorStorage};
        use std::sync::Arc;

        /// Helper to create a minimal SOTAQueryEngine for prompt tests.
        fn create_prompt_test_engine() -> SOTAQueryEngine {
            let vector_storage = Arc::new(MemoryVectorStorage::new("test", 384));
            let graph_storage = Arc::new(MemoryGraphStorage::new("test"));
            let embedding_provider: Arc<dyn crate::EmbeddingProvider> =
                Arc::new(MockProvider::default());
            let llm_provider: Arc<dyn crate::LLMProvider> = Arc::new(MockProvider::default());

            SOTAQueryEngine::new(
                SOTAQueryConfig::default(),
                vector_storage,
                graph_storage,
                embedding_provider,
                llm_provider,
            )
        }

        /// Helper to create a non-empty context for prompt testing.
        fn test_context() -> QueryContext {
            let mut ctx = QueryContext::default();
            ctx.chunks.push(RetrievedChunk::new(
                "chunk-1",
                "Rust is a systems programming language.",
                0.9,
            ));
            ctx
        }

        #[test]
        fn test_build_prompt_without_system_prompt() {
            let engine = create_prompt_test_engine();
            let context = test_context();

            let prompt = engine.build_prompt("What is Rust?", &context, None);

            assert!(prompt.contains("---Role---"));
            assert!(prompt.contains("---Instructions---"));
            assert!(prompt.contains("---Context---"));
            assert!(prompt.contains("What is Rust?"));
            // Should NOT contain additional instructions section
            assert!(!prompt.contains("---Additional Instructions---"));
        }

        #[test]
        fn test_build_prompt_with_system_prompt() {
            let engine = create_prompt_test_engine();
            let context = test_context();

            let prompt = engine.build_prompt(
                "What is Rust?",
                &context,
                Some("Always respond in French. Be concise."),
            );

            assert!(prompt.contains("---Role---"));
            assert!(prompt.contains("---Instructions---"));
            assert!(prompt.contains("---Additional Instructions---"));
            assert!(prompt.contains("Always respond in French. Be concise."));
            assert!(prompt.contains("---Context---"));
            assert!(prompt.contains("What is Rust?"));

            // Additional instructions should appear between instructions and context
            let instructions_pos = prompt.find("---Instructions---").unwrap();
            let additional_pos = prompt.find("---Additional Instructions---").unwrap();
            let context_pos = prompt.find("---Context---").unwrap();
            assert!(
                instructions_pos < additional_pos,
                "Additional instructions should come after base instructions"
            );
            assert!(
                additional_pos < context_pos,
                "Additional instructions should come before context"
            );
        }

        #[test]
        fn test_build_prompt_with_empty_system_prompt() {
            let engine = create_prompt_test_engine();
            let context = test_context();

            // Empty string should behave like None
            let prompt = engine.build_prompt("What is Rust?", &context, Some(""));
            assert!(!prompt.contains("---Additional Instructions---"));

            // Whitespace-only should also behave like None
            let prompt = engine.build_prompt("What is Rust?", &context, Some("   \n\t  "));
            assert!(!prompt.contains("---Additional Instructions---"));
        }

        #[test]
        fn test_build_prompt_empty_context() {
            let engine = create_prompt_test_engine();
            let empty_context = QueryContext::default();

            // Empty context should return a "no information" message regardless of system_prompt
            let prompt = engine.build_prompt("query", &empty_context, Some("Be concise"));
            assert!(prompt.contains("couldn't find any relevant information"));
            assert!(!prompt.contains("---Additional Instructions---"));
        }
    }
}
