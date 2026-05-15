//! Execute RAG query handler.
//!
//! @implements UC0201 (Execute Query)
//! @implements FEAT0007 (Multi-Mode Query Execution)
//! @implements FEAT0101-0106 (Query modes)

use axum::{extract::State, Json};
use tracing::{debug, error, warn};

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;
use crate::validation::validate_query;
use edgequake_query::{QueryMode, QueryRequest as EngineQueryRequest};

use super::{
    resolve_chunk_file_paths,
    workspace_resolve::{
        get_workspace, get_workspace_embedding_provider, get_workspace_llm_info,
        get_workspace_vector_storage,
    },
};
pub use crate::handlers::query_types::{
    ApprovedAlgorithmDto, ApprovedAlgorithmStepDto, QueryRequest, QueryResponse, QueryStats,
    ReferenceCodeSnippetDto, SourceReference,
};

/// Execute a RAG query with multi-mode retrieval.
///
/// # Implements
///
/// - **UC0201**: Execute Query
/// - **FEAT0007**: Multi-Mode Query Execution
/// - **FEAT0101**: Naive mode (vector search only)
/// - **FEAT0102**: Local mode (entity-centric)
/// - **FEAT0103**: Global mode (community summaries)
/// - **FEAT0104**: Hybrid mode (local + global)
/// - **FEAT0105**: Mix mode (adaptive blend)
/// - **FEAT0106**: Bypass mode (direct LLM, no RAG)
///
/// # Enforces
///
/// - **BR0101**: Token budget enforcement
/// - **BR0103**: Mode validation
/// - **BR0201**: Tenant/workspace scoping
///
/// # Returns
///
/// - `response`: LLM-generated answer
/// - `sources`: Source references with document lineage
/// - `stats`: Retrieval statistics (chunks, entities, latency)
#[utoipa::path(
    post,
    path = "/api/v1/query",
    tag = "Query",
    request_body = QueryRequest,
    responses(
        (status = 200, description = "Query executed successfully", body = QueryResponse),
        (status = 400, description = "Invalid query")
    )
)]
pub async fn execute_query(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Json(request): Json<QueryRequest>,
) -> ApiResult<Json<QueryResponse>> {
    debug!(
        tenant_id = ?tenant_ctx.tenant_id,
        workspace_id = ?tenant_ctx.workspace_id,
        query = %request.query,
        "Executing query with tenant context"
    );

    validate_query(&request.query, state.config.max_query_length)?;

    // Parse query mode
    let mode = request
        .mode
        .as_ref()
        .and_then(|m| QueryMode::parse(m))
        .unwrap_or(QueryMode::Hybrid);

    // Build engine query request with conversation history and tenant context
    let mut engine_request = EngineQueryRequest::new(&request.query).with_mode(mode);

    // SPEC-004: Thread system prompt extension if provided
    if let Some(ref system_prompt) = request.system_prompt {
        engine_request = engine_request.with_system_prompt(system_prompt);
    }

    // OODA-231.1: Fetch workspace to get correct tenant_id for data queries
    // WHY: Header tenant_id is for authentication (random UUID from frontend).
    // But the graph data was ingested with the workspace's actual tenant_id.
    // Using header tenant_id causes 0 results because of tenant_id mismatch.
    let workspace = if let Some(ref workspace_id) = tenant_ctx.workspace_id {
        get_workspace(&state, workspace_id).await.ok().flatten()
    } else {
        None
    };

    // Use workspace's tenant_id for data queries, fall back to header tenant_id
    let data_tenant_id = workspace
        .as_ref()
        .map(|ws| ws.tenant_id.to_string())
        .or_else(|| tenant_ctx.tenant_id.clone());

    if let Some(ref tenant_id) = data_tenant_id {
        engine_request = engine_request.with_tenant_id(tenant_id.clone());
    }
    if let Some(ref workspace_id) = tenant_ctx.workspace_id {
        engine_request = engine_request.with_workspace_id(workspace_id.clone());
    }

    // Workspace-scoped chunk_min_score override (None = engine default).
    if let Some(score) = workspace.as_ref().and_then(|ws| ws.chunk_min_score) {
        engine_request = engine_request.with_chunk_min_score(score);
    }

    // Workspace-scoped reranker strategy + model (None = engine default).
    if let Some(strategy) = workspace
        .as_ref()
        .and_then(|ws| ws.reranker_strategy.clone())
    {
        engine_request.reranker_strategy = Some(strategy);
    }
    if let Some(model) = workspace.as_ref().and_then(|ws| ws.reranker_model.clone()) {
        engine_request.reranker_model = Some(model);
    }

    // Workspace-scoped Qwen3-Embedding query instruction (None = engine default).
    if let Some(instruction) = workspace
        .as_ref()
        .and_then(|ws| ws.embedding_query_instruction.clone())
    {
        engine_request = engine_request.with_query_instruction(instruction);
    }

    if request.context_only {
        engine_request = engine_request.context_only();
    }

    if request.prompt_only {
        engine_request = engine_request.prompt_only();
    }

    // SPEC-032: Add LLM provider/model overrides if provided in request
    // This allows query-time override of the LLM provider and model
    if let Some(ref provider) = request.llm_provider {
        debug!(provider = %provider, "Using LLM provider override from request");
        engine_request = engine_request.with_llm_provider(provider);
    }
    if let Some(ref model) = request.llm_model {
        debug!(model = %model, "Using LLM model override from request");
        engine_request = engine_request.with_llm_model(model);
    }

    // Add conversation history if provided
    if let Some(history) = &request.conversation_history {
        let engine_history: Vec<edgequake_query::ConversationMessage> = history
            .iter()
            .map(|m| edgequake_query::ConversationMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();
        engine_request = engine_request.with_conversation_history(engine_history);
    }

    // SPEC-005: Resolve document filter → allowed_document_ids
    if let Some(ref filter) = request.document_filter {
        if let Some(allowed_ids) = super::document_filter_resolver::resolve_document_filter(
            state.kv_storage.as_ref(),
            filter,
            &data_tenant_id,
            &tenant_ctx.workspace_id,
        )
        .await?
        {
            debug!(
                matched_doc_count = allowed_ids.len(),
                "Document filter resolved — restricting query scope"
            );
            engine_request = engine_request.with_allowed_document_ids(allowed_ids);
        }
    }

    // SPEC-032 & SPEC-033: Get workspace-specific embedding provider AND vector storage
    // If workspace has custom embedding config, use workspace-specific resources
    let result = if let Some(ref workspace_id) = tenant_ctx.workspace_id {
        // Try to get workspace embedding and vector storage configuration
        let embedding_result = get_workspace_embedding_provider(&state, workspace_id).await;
        let vector_result = get_workspace_vector_storage(&state, workspace_id).await;

        // Check if LLM provider override is requested (from request or workspace config)
        let llm_override = if let (Some(ref provider), Some(ref model)) =
            (&request.llm_provider, &request.llm_model)
        {
            // Case 1: Explicit provider/model in request
            debug!(provider = %provider, model = %model, "Creating LLM provider override from request");
            Some(
                crate::safety_limits::create_safe_llm_provider(provider, model).map_err(|e| {
                    ApiError::Internal(format!("Failed to create LLM provider: {}", e))
                })?,
            )
        } else if let Some(ref ws) = workspace {
            // Case 2: Use workspace LLM config (same as streaming endpoint)
            if !ws.llm_provider.is_empty() && !ws.llm_model.is_empty() {
                debug!(
                    provider = %ws.llm_provider,
                    model = %ws.llm_model,
                    "Creating LLM provider override from workspace config"
                );
                match crate::safety_limits::create_safe_llm_provider(
                    &ws.llm_provider,
                    &ws.llm_model,
                ) {
                    Ok(provider) => Some(provider),
                    Err(e) => {
                        warn!(
                            provider = %ws.llm_provider,
                            model = %ws.llm_model,
                            error = %e,
                            "Workspace LLM provider failed, falling back to server default"
                        );
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        match (embedding_result, vector_result) {
            (Ok(Some(embedding_provider)), Ok(Some(vector_storage))) => {
                // Full workspace isolation: use both workspace-specific embedding and vector storage
                debug!(
                    workspace_id = %workspace_id,
                    has_llm_override = llm_override.is_some(),
                    "Using workspace-specific embedding provider AND vector storage for query"
                );
                state
                    .sota_engine
                    .query_with_full_config(
                        engine_request,
                        embedding_provider,
                        vector_storage,
                        llm_override,
                    )
                    .await
                    .map_err(|e| ApiError::Internal(format!("Query failed: {}", e)))?
            }
            (Ok(Some(embedding_provider)), _) => {
                // Workspace-specific embedding only
                debug!(
                    workspace_id = %workspace_id,
                    "Using workspace-specific embedding provider for query"
                );
                state
                    .sota_engine
                    .query_with_embedding_provider(engine_request, embedding_provider)
                    .await
                    .map_err(|e| ApiError::Internal(format!("Query failed: {}", e)))?
            }
            (Ok(None), Ok(Some(vector_storage))) => {
                // Workspace uses default embedding model but has its own vector storage table.
                // WHY: Injection (SPEC-0002) and document ingestion (SPEC-033) store vectors in
                // the workspace-specific table.  We must search that table even when the
                // workspace shares the server's default embedding provider.
                debug!(
                    workspace_id = %workspace_id,
                    "Using default embedding + workspace-specific vector storage for query"
                );
                state
                    .sota_engine
                    .query_with_vector_storage(engine_request, vector_storage)
                    .await
                    .map_err(|e| ApiError::Internal(format!("Query failed: {}", e)))?
            }
            (Ok(None), _) => {
                // No workspace-specific config and no workspace vector storage — use defaults.
                debug!(
                    workspace_id = %workspace_id,
                    "Using default embedding provider for query (no workspace vector storage)"
                );
                state
                    .sota_engine
                    .query(engine_request)
                    .await
                    .map_err(|e| ApiError::Internal(format!("Query failed: {}", e)))?
            }
            (Err(e), _) => {
                // OODA-229: Return configuration errors to the user instead of silent fallback
                // WHY: If workspace is configured for OpenAI but API key is missing, using
                // the default provider would return wrong results (different embeddings).
                // Better to fail fast with a clear error message.
                if matches!(&e, ApiError::ConfigError(_)) {
                    error!(
                        workspace_id = %workspace_id,
                        error = %e,
                        "Workspace embedding configuration error - returning to user"
                    );
                    return Err(e);
                }

                // For other errors, fallback to default with warning
                warn!(
                    workspace_id = %workspace_id,
                    error = %e,
                    "Failed to get workspace embedding config, using default"
                );
                state
                    .sota_engine
                    .query(engine_request)
                    .await
                    .map_err(|e| ApiError::Internal(format!("Query failed: {}", e)))?
            }
        }
    } else {
        // No workspace context, use default engine embedding
        state
            .sota_engine
            .query(engine_request)
            .await
            .map_err(|e| ApiError::Internal(format!("Query failed: {}", e)))?
    };

    // Convert sources from context
    let mut sources = Vec::new();

    let mut ref_counter = 1usize;
    let mut chunk_sources: Vec<SourceReference> = result
        .context
        .chunks
        .iter()
        .map(|chunk| {
            let ref_id = ref_counter;
            ref_counter += 1;

            SourceReference {
                source_type: "chunk".to_string(),
                id: chunk.id.clone(),
                score: chunk.score,
                snippet: Some(if request.context_only {
                    chunk.content.clone()
                } else {
                    chunk.content.chars().take(200).collect()
                }),
                reference_id: Some(ref_id),
                document_id: chunk.document_id.clone(),
                file_path: None, // Resolved below via KV metadata lookup
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                chunk_index: chunk.chunk_index,
                kind: chunk.kind.clone(),
                figure_id: chunk.figure_id.clone(),
                caption: chunk.caption.clone(),
                entity_type: None,
                degree: None,
                source_chunk_ids: None,
                ..Default::default()
            }
        })
        .collect();

    // Resolve document_id → file_path (document title) for chunk sources
    resolve_chunk_file_paths(state.kv_storage.as_ref(), &mut chunk_sources).await;

    // SPEC-0002: Exclude injection artifacts from cited sources.
    // Injection chunks enrich LLM context but must NOT appear as source citations.
    chunk_sources.retain(|s| {
        !s.document_id
            .as_deref()
            .unwrap_or("")
            .starts_with("injection::")
    });

    sources.extend(chunk_sources);

    for entity in &result.context.entities {
        // SPEC-0002: Skip injection-only entities from citations.
        // Injection source_document_id is "injection::{workspace_id}::{id}".
        // These entities enrich graph context but must not be cited as sources.
        if entity
            .source_document_id
            .as_deref()
            .unwrap_or("")
            .starts_with("injection::")
        {
            continue;
        }

        let ref_id = ref_counter;
        ref_counter += 1;

        sources.push(SourceReference {
            source_type: "entity".to_string(),
            id: entity.name.clone(),
            score: entity.score,
            snippet: Some(if request.context_only {
                entity.description.clone()
            } else {
                entity.description.chars().take(200).collect()
            }),
            reference_id: Some(ref_id),
            document_id: entity.source_document_id.clone(),
            file_path: entity.source_file_path.clone(),
            start_line: None,
            end_line: None,
            chunk_index: None,
            // SPEC-006: Enrich entity metadata
            entity_type: Some(entity.entity_type.clone()),
            degree: if entity.degree > 0 {
                Some(entity.degree)
            } else {
                None
            },
            source_chunk_ids: if entity.source_chunk_ids.is_empty() {
                None
            } else {
                Some(entity.source_chunk_ids.clone())
            },
            ..Default::default()
        });
    }

    for rel in &result.context.relationships {
        // SPEC-0002: Skip injection-only relationships from citations.
        if rel
            .source_document_id
            .as_deref()
            .unwrap_or("")
            .starts_with("injection::")
        {
            continue;
        }

        let ref_id = ref_counter;
        ref_counter += 1;

        sources.push(SourceReference {
            source_type: "relationship".to_string(),
            id: format!("{}->{}", rel.source, rel.target),
            score: rel.score,
            snippet: Some(format!(
                "{} {} {}",
                rel.source, rel.relation_type, rel.target
            )),
            reference_id: Some(ref_id),
            document_id: rel.source_document_id.clone(),
            file_path: rel.source_file_path.clone(),
            start_line: None,
            end_line: None,
            chunk_index: None,
            entity_type: None,
            degree: None,
            source_chunk_ids: None,
            ..Default::default()
        });
    }

    // Generate conversation ID if conversation history was provided
    let conversation_id = if request.conversation_history.is_some() {
        Some(uuid::Uuid::new_v4().to_string())
    } else {
        None
    };

    // SPEC-032 Item 18, 22: Get LLM provider/model info for lineage tracking
    let (llm_provider, llm_model) =
        get_workspace_llm_info(&state, tenant_ctx.workspace_id.as_deref()).await;

    // SPEC-032 Item 18: Calculate tokens per second
    let tokens_used = if result.stats.generated_tokens > 0 {
        Some(result.stats.generated_tokens)
    } else {
        None
    };

    let tokens_per_second =
        if result.stats.generation_time_ms > 0 && result.stats.generated_tokens > 0 {
            Some(
                (result.stats.generated_tokens as f32) / (result.stats.generation_time_ms as f32)
                    * 1000.0,
            )
        } else {
            None
        };

    // Phase 1 Reference Code GraphRAG: lift approved code snippets out of the
    // context into a first-class response field so callers that render their
    // own output (e.g. OpenWebUI tool) don't have to grep the LLM prompt.
    let reference_code: Vec<ReferenceCodeSnippetDto> = result
        .context
        .reference_code
        .iter()
        .map(|s| ReferenceCodeSnippetDto {
            algorithm_id: s.algorithm_id.clone(),
            algorithm_name: s.algorithm_name.clone(),
            document_id: s.document_id.clone(),
            file_path: s.file_path.clone(),
            start_line: s.start_line,
            end_line: s.end_line,
            language: s.language.clone(),
            snippet: s.snippet.clone(),
            repo_url: s.repo_url.clone(),
            repo_commit: s.repo_commit.clone(),
            match_rationale: s.match_rationale.clone(),
        })
        .collect();

    // Symmetric with reference_code: surface reviewer-approved algorithm
    // definitions as their own top-level field. Clients that render chat
    // output themselves (OpenWebUI tool, MCP tool) can show pseudocode +
    // steps directly rather than parsing it out of the context string.
    let approved_algorithms: Vec<ApprovedAlgorithmDto> = result
        .context
        .approved_algorithms
        .iter()
        .map(|a| ApprovedAlgorithmDto {
            algorithm_id: a.algorithm_id.clone(),
            document_id: a.document_id.clone(),
            name: a.name.clone(),
            algorithm_type: a.algorithm_type.clone(),
            description: a.description.clone(),
            pseudocode: a.pseudocode.clone(),
            complexity: a.complexity.clone(),
            steps: a
                .steps
                .iter()
                .map(|s| ApprovedAlgorithmStepDto {
                    number: s.number,
                    action: s.action.clone(),
                    details: s.details.clone(),
                })
                .collect(),
            tags: a.tags.clone(),
            confidence: a.confidence.clone(),
        })
        .collect();

    let response = QueryResponse {
        answer: result.answer,
        mode: result.mode.to_string(),
        sources,
        stats: QueryStats {
            embedding_time_ms: result.stats.embedding_time_ms,
            retrieval_time_ms: result.stats.retrieval_time_ms,
            generation_time_ms: result.stats.generation_time_ms,
            total_time_ms: result.stats.total_time_ms,
            sources_retrieved: result.context.chunks.len()
                + result.context.entities.len()
                + result.context.relationships.len(),
            // SPEC-032 Item 18, 22: Token metrics and model lineage
            tokens_used,
            tokens_per_second,
            llm_provider,
            llm_model,
        },
        conversation_id,
        reference_code,
        approved_algorithms,
    };

    Ok(Json(response))
}
