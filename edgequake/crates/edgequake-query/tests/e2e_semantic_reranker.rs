//! E2E test for the semantic (HTTP) reranker registry.
//!
//! Verifies that:
//! - `with_semantic_config` wires the engine with a working registry
//! - `semantic_reranker_for_model(None)` resolves the env default model
//! - `semantic_reranker_for_model(Some("..."))` overrides the model
//! - Per-model cache returns the same instance on repeat calls
//! - The cached reranker actually hits the configured `/v1/rerank` endpoint
//!   and returns sensible scores
//!
//! Marked `#[ignore]` — requires a running rerank server:
//! ```
//! curl http://127.0.0.1:8060/v1/models  # confirm jina-reranker-v3 listed
//! cargo test -p edgequake-query --test e2e_semantic_reranker -- --ignored --nocapture
//! ```

use std::sync::Arc;
use std::time::Duration;

use edgequake_llm::{EmbeddingProvider, MockProvider};
use edgequake_query::sota_engine::SemanticRerankerConfig;
use edgequake_query::{SOTAQueryConfig, SOTAQueryEngine};
use edgequake_storage::{MemoryGraphStorage, MemoryVectorStorage};

fn semantic_url() -> String {
    std::env::var("RERANKER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8060/v1/rerank".to_string())
}

async fn build_engine(default_model: &str) -> SOTAQueryEngine {
    let dimension = MockProvider::new().dimension();
    let vector_storage = Arc::new(MemoryVectorStorage::new("rerank-test", dimension));
    let graph_storage = Arc::new(MemoryGraphStorage::new("rerank-test-graph"));
    let provider = Arc::new(MockProvider::new());

    SOTAQueryEngine::with_mock_keywords(
        SOTAQueryConfig::default(),
        vector_storage,
        graph_storage,
        provider.clone(),
        provider,
    )
    .with_semantic_config(SemanticRerankerConfig {
        default_model: default_model.to_string(),
        base_url: semantic_url(),
        api_key: None,
        timeout: Duration::from_secs(30),
    })
}

#[tokio::test]
#[ignore]
async fn registry_returns_default_model_when_override_is_none() {
    let engine = build_engine("jina-reranker-v3").await;

    let r = engine
        .semantic_reranker_for_model(None)
        .await
        .expect("semantic_config wired but resolver returned None");
    assert_eq!(r.model(), "jina-reranker-v3");
    assert_eq!(r.name(), "http");
}

#[tokio::test]
#[ignore]
async fn registry_honors_per_request_model_override() {
    let engine = build_engine("jina-reranker-v3").await;

    let r = engine
        .semantic_reranker_for_model(Some("bge-reranker-v2-m3"))
        .await
        .expect("resolver returned None for model override");
    assert_eq!(
        r.model(),
        "bge-reranker-v2-m3",
        "override must propagate to HttpReranker config"
    );
}

#[tokio::test]
#[ignore]
async fn registry_caches_per_model_instance() {
    let engine = build_engine("jina-reranker-v3").await;

    let a = engine.semantic_reranker_for_model(None).await.unwrap();
    let b = engine.semantic_reranker_for_model(None).await.unwrap();
    assert!(
        Arc::ptr_eq(&a, &b),
        "second call for same model must return cached Arc, not a fresh instance"
    );

    let other = engine
        .semantic_reranker_for_model(Some("bge-reranker-v2-m3"))
        .await
        .unwrap();
    assert!(
        !Arc::ptr_eq(&a, &other),
        "different model name must produce a distinct cached instance"
    );
}

#[tokio::test]
#[ignore]
async fn end_to_end_rerank_against_llama_swap() {
    let engine = build_engine("jina-reranker-v3").await;

    let reranker = engine
        .semantic_reranker_for_model(None)
        .await
        .expect("resolver returned None despite semantic_config wired");

    let docs = vec![
        "Paris is the capital and largest city of France.".to_string(),
        "Berlin is the capital of Germany.".to_string(),
        "The Eiffel Tower is in Paris.".to_string(),
        "Cooking pasta requires boiling water.".to_string(),
    ];

    let results = reranker
        .rerank("What is the capital of France?", &docs, Some(4))
        .await
        .expect("rerank request to live llama-swap failed");

    assert_eq!(results.len(), 4, "expected scores for all 4 docs");
    eprintln!("rerank results:");
    for r in &results {
        eprintln!("  doc[{}] = {:.4}", r.index, r.relevance_score);
    }

    // doc 0 (Paris is capital) must outrank doc 3 (cooking) — that's the
    // whole point of a cross-encoder reranker.
    let paris = results.iter().find(|r| r.index == 0).unwrap();
    let cooking = results.iter().find(|r| r.index == 3).unwrap();
    assert!(
        paris.relevance_score > cooking.relevance_score,
        "Paris-capital doc must score higher than the cooking doc — \
         got paris={:.4} vs cooking={:.4}",
        paris.relevance_score,
        cooking.relevance_score,
    );

    // Sigmoid normalization: when the engine builds the HttpReranker for a
    // non-cloud `base_url` (e.g. llama-swap at 127.0.0.1), `sigmoid_normalize`
    // is set so raw classifier logits get mapped to (0, 1). The engine-wide
    // `min_rerank_score` floor is tuned for that range — if normalization
    // ever regresses, every chunk drops below the floor and the rerank
    // becomes a no-op. Catch that here.
    for r in &results {
        assert!(
            r.relevance_score >= 0.0 && r.relevance_score <= 1.0,
            "score out of [0,1] — sigmoid normalization regressed: \
             doc[{}] = {:.4}",
            r.index,
            r.relevance_score,
        );
    }
}

#[tokio::test]
#[ignore]
async fn nonexistent_model_propagates_hard_error() {
    let engine = build_engine("jina-reranker-v3").await;

    let reranker = engine
        .semantic_reranker_for_model(Some("definitely-not-a-real-model-xyz"))
        .await
        .unwrap();

    let result = reranker
        .rerank("query", &["doc".to_string()], Some(1))
        .await;

    assert!(
        result.is_err(),
        "unknown model must produce a hard error (404 / model-not-found from llama-swap), \
         not a silent zero-score fallback. got: {result:?}",
    );
}
