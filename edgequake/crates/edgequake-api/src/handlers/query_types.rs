//! Query DTO types.
//!
//! This module contains all Data Transfer Objects for the query API.
//! Extracted from query.rs for modularity and single responsibility.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// ============================================================================
// Request DTOs
// ============================================================================

/// Document filter criteria for narrowing query scope.
///
/// Allows filtering RAG query results to only include content from documents
/// matching the specified date range and/or name pattern.
///
/// @implements SPEC-005: Document date and pattern filters
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct DocumentFilter {
    /// Start date (inclusive) in ISO 8601 format (e.g., "2025-01-01T00:00:00Z").
    /// Only documents created on or after this date are included.
    #[serde(default)]
    pub date_from: Option<String>,

    /// End date (inclusive) in ISO 8601 format (e.g., "2025-12-31T23:59:59Z").
    /// Only documents created on or before this date are included.
    #[serde(default)]
    pub date_to: Option<String>,

    /// Case-insensitive substring pattern to match against document titles.
    /// Comma-separated values are treated as OR conditions.
    /// Example: "report,summary" matches documents containing "report" OR "summary".
    #[serde(default)]
    pub document_pattern: Option<String>,
}

/// A single message in the conversation history.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ConversationMessage {
    /// Role of the message sender (user or assistant).
    pub role: String,

    /// Content of the message.
    pub content: String,
}

/// Query request.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct QueryRequest {
    /// The query text.
    pub query: String,

    /// Query mode (naive, local, global, hybrid, mix).
    #[serde(default)]
    pub mode: Option<String>,

    /// Only return context, don't generate an answer.
    #[serde(default)]
    pub context_only: bool,

    /// Return the formatted prompt instead of calling the LLM.
    /// Useful for debugging or using your own LLM.
    #[serde(default)]
    pub prompt_only: bool,

    /// Include detailed reference metadata (document_id, file_path, reference_id) in sources.
    #[serde(default)]
    pub include_references: bool,

    /// Maximum number of results.
    #[serde(default)]
    pub max_results: Option<usize>,

    /// Conversation history for multi-turn context.
    #[serde(default)]
    pub conversation_history: Option<Vec<ConversationMessage>>,

    /// LLM provider to use for this query (e.g., "openai", "ollama", "lmstudio").
    /// If not provided, uses the workspace or server default.
    /// @implements SPEC-032: Provider selection in query interface
    #[serde(default)]
    pub llm_provider: Option<String>,

    /// Specific model name within the provider (e.g., "gpt-4o-mini", "gemma3:12b").
    /// When combined with provider, allows full model selection from models.toml.
    /// If not provided, uses the provider's default chat model.
    /// @implements SPEC-032: Full model selection in query interface
    #[serde(default)]
    pub llm_model: Option<String>,

    /// Optional system prompt extension injected into the LLM prompt.
    /// Extends (not replaces) the base RAG prompt with additional instructions.
    /// @implements SPEC-004: System prompt extension point
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Optional document filter to narrow query scope by date range or name pattern.
    /// When set, only chunks/entities from matching documents are used in retrieval.
    /// @implements SPEC-005: Document date and pattern filters
    #[serde(default)]
    pub document_filter: Option<DocumentFilter>,
}

/// Streaming query request.
///
/// @implements SPEC-006: Unified streaming protocol
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct StreamQueryRequest {
    /// The query text.
    pub query: String,

    /// Query mode.
    #[serde(default)]
    pub mode: Option<String>,

    /// Optional system prompt extension injected into the LLM prompt.
    /// @implements SPEC-004: System prompt extension point
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Optional document filter to narrow query scope by date range or name pattern.
    /// @implements SPEC-005 + SPEC-006: Document filters for streaming queries
    #[serde(default)]
    pub document_filter: Option<DocumentFilter>,

    /// LLM provider to use for this query (e.g., "openai", "ollama", "lmstudio").
    /// @implements SPEC-006 + SPEC-032: Provider selection in streaming queries
    #[serde(default)]
    pub llm_provider: Option<String>,

    /// Specific model name within the provider.
    /// @implements SPEC-006 + SPEC-032: Model selection in streaming queries
    #[serde(default)]
    pub llm_model: Option<String>,

    /// Stream format version: "v1" (raw text) or "v2" (structured JSON events, default).
    /// @implements SPEC-006: Backward compatibility
    #[serde(default)]
    pub stream_format: Option<String>,
}

// ============================================================================
// Streaming Event Types (SPEC-006)
// ============================================================================

/// Streaming SSE event types for the query endpoint.
///
/// @implements SPEC-006: Unified streaming protocol for /query/stream
#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryStreamEvent {
    /// Context/sources retrieved before generation starts.
    Context {
        sources: Vec<SourceReference>,
        query_mode: String,
        retrieval_time_ms: u64,
    },

    /// Token generated during LLM streaming.
    Token { content: String },

    /// Chain-of-thought reasoning content.
    Thinking { content: String },

    /// Stream complete — includes full statistics.
    Done {
        stats: QueryStreamStats,
        #[serde(skip_serializing_if = "Option::is_none")]
        llm_provider: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        llm_model: Option<String>,
    },

    /// Error occurred during streaming.
    Error { message: String, code: String },
}

/// Statistics emitted in the `done` event.
///
/// @implements SPEC-006 FR-003: Retrieval statistics in streaming events
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct QueryStreamStats {
    /// Embedding time in ms.
    pub embedding_time_ms: u64,

    /// Retrieval time in ms.
    pub retrieval_time_ms: u64,

    /// Generation time in ms.
    pub generation_time_ms: u64,

    /// Total time in ms.
    pub total_time_ms: u64,

    /// Number of sources retrieved.
    pub sources_retrieved: usize,

    /// Tokens used for generation.
    pub tokens_used: u32,

    /// Tokens per second generation speed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f32>,

    /// Query mode used (after adaptive selection).
    pub query_mode: String,
}

// ============================================================================
// Response DTOs
// ============================================================================

/// Query response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct QueryResponse {
    /// Generated answer.
    pub answer: String,

    /// Query mode used.
    pub mode: String,

    /// Retrieved context sources.
    pub sources: Vec<SourceReference>,

    /// Query statistics.
    pub stats: QueryStats,

    /// Conversation ID for multi-turn context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,

    /// Approved reference-code snippets attached to the query via the
    /// Reference Code GraphRAG enrichment (Phase 1). Empty unless the
    /// workspace has approved code_artifacts whose embedding is close to
    /// the query vector. Separate from `sources[]` because chunks and
    /// code snippets have different renderers and different lifetimes.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub reference_code: Vec<ReferenceCodeSnippetDto>,

    /// Reviewer-approved algorithm definitions (name + description + steps
    /// + pseudocode) matched to the query via semantic search. Separate
    /// from `sources[]` because structured algorithm data has its own
    /// renderer — treating it as a plain text chunk would lose the step
    /// structure.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub approved_algorithms: Vec<ApprovedAlgorithmDto>,
}

/// One step in an [`ApprovedAlgorithmDto`].
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ApprovedAlgorithmStepDto {
    pub number: usize,
    pub action: String,
    pub details: String,
}

/// Public shape of an [`ApprovedAlgorithmSnippet`] for the API. Mirrors
/// `edgequake_query::context::ApprovedAlgorithmSnippet` so the api crate
/// doesn't re-export query types in its JSON contract.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ApprovedAlgorithmDto {
    pub algorithm_id: String,
    pub document_id: String,
    pub name: String,
    pub algorithm_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pseudocode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub complexity: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub steps: Vec<ApprovedAlgorithmStepDto>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tags: Vec<String>,
    pub confidence: String,
}

/// Public shape of a [`ReferenceCodeSnippet`] for the API.
///
/// Mirrors `edgequake_query::context::ReferenceCodeSnippet` but declared
/// locally so `edgequake-api` doesn't re-export query-crate types in its
/// public JSON contract.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ReferenceCodeSnippetDto {
    pub algorithm_id: String,
    pub algorithm_name: String,
    pub document_id: String,
    pub file_path: String,
    pub start_line: i32,
    pub end_line: i32,
    pub language: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_url: Option<String>,
    pub repo_commit: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_rationale: Option<String>,
}

/// A source reference.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct SourceReference {
    /// Source type (chunk, entity, relationship).
    pub source_type: String,

    /// Source ID.
    pub id: String,

    /// Relevance score.
    pub score: f32,

    /// Content snippet.
    pub snippet: Option<String>,

    /// Reference ID for citation (1, 2, 3, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_id: Option<usize>,

    /// Document ID that this reference came from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,

    /// Original file path of the source document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,

    /// Start line number in the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_line: Option<usize>,

    /// End line number in the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,

    /// Chunk index in the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_index: Option<usize>,

    // ========================================================================
    // SPEC-006: Entity metadata enrichment (FR-002)
    // ========================================================================
    /// Entity type (e.g., "PERSON", "ORGANIZATION"). Only set for source_type="entity".
    /// @implements SPEC-006: Entity metadata enrichment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_type: Option<String>,

    /// Number of graph connections. Only set for source_type="entity".
    /// @implements SPEC-006: Entity degree in source references
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degree: Option<usize>,

    /// Source chunk IDs where entity was mentioned (provenance). Only set for source_type="entity".
    /// @implements SPEC-006: Entity provenance tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_chunk_ids: Option<Vec<String>>,
}

/// Query statistics.
///
/// @implements SPEC-032 Item 18, 22: Token metrics and model lineage
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct QueryStats {
    /// Embedding time in ms.
    pub embedding_time_ms: u64,

    /// Retrieval time in ms.
    pub retrieval_time_ms: u64,

    /// Generation time in ms.
    pub generation_time_ms: u64,

    /// Total time in ms.
    pub total_time_ms: u64,

    /// Number of sources retrieved.
    pub sources_retrieved: usize,

    // ========================================================================
    // SPEC-032: Token metrics and model lineage (Items 18, 22)
    // ========================================================================
    /// Number of tokens generated in the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_used: Option<usize>,

    /// Tokens per second generation speed (calculated as tokens_used / generation_time_ms * 1000).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f32>,

    /// LLM provider used for generation (e.g., "ollama", "openai", "lmstudio").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_provider: Option<String>,

    /// LLM model name used for generation (e.g., "gemma3:12b", "gpt-4o-mini").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_model: Option<String>,
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_request_minimal() {
        let json = r#"{"query": "What is RAG?"}"#;
        let req: QueryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "What is RAG?");
        assert!(!req.context_only);
        assert!(!req.prompt_only);
    }

    #[test]
    fn test_query_request_full() {
        let json = r#"{
            "query": "What is AI?",
            "mode": "hybrid",
            "context_only": true,
            "include_references": true,
            "max_results": 10
        }"#;
        let req: QueryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.mode, Some("hybrid".to_string()));
        assert!(req.context_only);
        assert!(req.include_references);
    }

    #[test]
    fn test_conversation_message() {
        let json = r#"{"role": "user", "content": "Hello"}"#;
        let msg: ConversationMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_stream_query_request() {
        let json = r#"{"query": "Tell me about embeddings", "mode": "local"}"#;
        let req: StreamQueryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "Tell me about embeddings");
        assert_eq!(req.mode, Some("local".to_string()));
    }

    #[test]
    fn test_source_reference_serialization() {
        let source = SourceReference {
            source_type: "chunk".to_string(),
            id: "chunk_123".to_string(),
            score: 0.95,
            snippet: Some("This is a test snippet".to_string()),
            reference_id: Some(1),
            document_id: Some("doc_456".to_string()),
            file_path: Some("docs/test.md".to_string()),
            start_line: Some(10),
            end_line: Some(20),
            chunk_index: Some(2),
            entity_type: None,
            degree: None,
            source_chunk_ids: None,
        };
        let json = serde_json::to_value(&source).unwrap();
        assert_eq!(json["source_type"], "chunk");
        // Use approximate comparison for floats
        let score = json["score"].as_f64().unwrap();
        assert!((score - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_source_reference_minimal() {
        let source = SourceReference {
            source_type: "entity".to_string(),
            id: "ENT_ABC".to_string(),
            score: 0.8,
            snippet: None,
            reference_id: None,
            document_id: None,
            file_path: None,
            start_line: None,
            end_line: None,
            chunk_index: None,
            entity_type: Some("ORGANIZATION".to_string()),
            degree: Some(5),
            source_chunk_ids: Some(vec!["chunk-1".to_string()]),
        };
        let json = serde_json::to_value(&source).unwrap();
        assert!(json.get("reference_id").is_none());
        // SPEC-006: Verify entity metadata fields are serialized
        assert_eq!(json["entity_type"], "ORGANIZATION");
        assert_eq!(json["degree"], 5);
        assert_eq!(json["source_chunk_ids"], serde_json::json!(["chunk-1"]));
    }

    #[test]
    fn test_query_stats_serialization() {
        let stats = QueryStats {
            embedding_time_ms: 50,
            retrieval_time_ms: 100,
            generation_time_ms: 500,
            total_time_ms: 650,
            sources_retrieved: 5,
            // SPEC-032 Item 18, 22: Token metrics and model lineage
            tokens_used: Some(124),
            tokens_per_second: Some(248.0),
            llm_provider: Some("ollama".to_string()),
            llm_model: Some("gemma3:12b".to_string()),
        };
        let json = serde_json::to_value(&stats).unwrap();
        assert_eq!(json["total_time_ms"], 650);
        assert_eq!(json["sources_retrieved"], 5);
        // SPEC-032: Verify new fields
        assert_eq!(json["tokens_used"], 124);
        assert_eq!(json["tokens_per_second"], 248.0);
        assert_eq!(json["llm_provider"], "ollama");
        assert_eq!(json["llm_model"], "gemma3:12b");
    }

    #[test]
    fn test_query_response_serialization() {
        let response = QueryResponse {
            answer: "RAG is Retrieval Augmented Generation".to_string(),
            mode: "hybrid".to_string(),
            sources: vec![],
            stats: QueryStats {
                embedding_time_ms: 10,
                retrieval_time_ms: 20,
                generation_time_ms: 100,
                total_time_ms: 130,
                sources_retrieved: 0,
                // SPEC-032 Item 18, 22: Token metrics and model lineage (optional in test)
                tokens_used: None,
                tokens_per_second: None,
                llm_provider: None,
                llm_model: None,
            },
            conversation_id: None,
            reference_code: Vec::new(),
            approved_algorithms: Vec::new(),
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["mode"], "hybrid");
        assert!(json.get("conversation_id").is_none());
        assert!(json.get("approved_algorithms").is_none()); // skip_serializing_if (empty)
    }
}
