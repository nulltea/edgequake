//! Query execution handlers.
//!
//! @implements FEAT0403
//! @implements SPEC-032: Workspace-specific embedding in query process
//!
//! # Implements
//!
//! - **UC0201**: Execute Query
//! - **UC0202**: Query with Conversation History
//! - **UC0203**: Stream Query Response
//! - **FEAT0403**: Query Execution Endpoint
//! - **FEAT0404**: Query Streaming Endpoint
//! - **FEAT0007**: Multi-Mode Query Execution
//! - **FEAT0101-0106**: Query modes (naive/local/global/hybrid/mix/bypass)
//!
//! # Enforces
//!
//! - **BR0101**: Token budget must not exceed LLM context window
//! - **BR0103**: Query mode must be valid enum value
//! - **BR0105**: Empty queries are rejected
//! - **BR0201**: Tenant isolation (queries scoped to workspace)
//!
//! # Workspace-Specific Embedding (SPEC-032)
//!
//! Queries use the embedding model configured for the workspace. This allows:
//! - Different workspaces to use different embedding providers (OpenAI, Ollama, LM Studio)
//! - Dimension-specific vector search per workspace
//!
//! # Endpoints
//!
//! | Method | Path | Handler | Description |
//! |--------|------|---------|-------------|
//! | POST | `/api/v1/query` | [`execute_query`] | Execute RAG query |
//! | POST | `/api/v1/query/stream` | [`execute_query_stream`] | Stream query response |
//!
//! # Query Flow
//!
//! ```text
//! POST /api/v1/query
//!        ↓
//!   Validate query length
//!        ↓
//!   Parse mode (default: hybrid)
//!        ↓
//!   Add tenant context (BR0201)
//!        ↓
//!   Load workspace embedding config (SPEC-032)
//!        ↓
//!   Execute via SOTA engine with workspace embedding
//!        ↓
//!   Format response + sources
//! ```

pub(crate) mod document_filter_resolver;
mod query_execute;
mod query_stream;
pub(crate) mod workspace_resolve;

pub use query_execute::*;
pub use query_stream::*;

// Re-export DTOs for backward compatibility
pub use crate::handlers::query_types::{
    ConversationMessage, QueryRequest, QueryResponse, QueryStats, SourceReference,
    StreamQueryRequest,
};

use std::collections::{HashMap, HashSet};

use tracing::{debug, warn};

use crate::handlers::query_types::SourceReference as SourceRef;

// ============================================================================
// Shared Helper Functions
// ============================================================================

/// Resolve document IDs to document titles from KV metadata.
///
/// Looks up `"{document_id}-metadata"` in KV storage for each unique document ID,
/// extracting the `"title"` field (falling back to `"file_name"`).
/// Returns a HashMap mapping document_id → document title.
async fn resolve_document_names(
    kv_storage: &dyn edgequake_storage::traits::KVStorage,
    document_ids: &[String],
) -> HashMap<String, String> {
    if document_ids.is_empty() {
        return HashMap::new();
    }

    // Deduplicate — multiple chunks often reference the same document
    let unique_ids: Vec<String> = document_ids
        .iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .cloned()
        .collect();

    let mut doc_names = HashMap::new();

    for doc_id in &unique_ids {
        let metadata_key = format!("{}-metadata", doc_id);
        match kv_storage.get_by_id(&metadata_key).await {
            Ok(Some(metadata)) => {
                if let Some(title) = metadata
                    .get("title")
                    .or_else(|| metadata.get("file_name"))
                    .and_then(|v| v.as_str())
                {
                    doc_names.insert(doc_id.clone(), title.to_string());
                }
            }
            Ok(None) => {
                debug!(document_id = %doc_id, "No metadata found for document");
            }
            Err(e) => {
                warn!(document_id = %doc_id, error = %e, "Failed to fetch document metadata");
            }
        }
    }

    doc_names
}

/// Resolve `file_path` for chunk sources that are missing it.
///
/// Collects unique document IDs from chunk sources, performs a batched KV lookup
/// for document metadata, and patches `file_path` with the resolved document title.
/// Non-chunk sources and sources that already have `file_path` are left unchanged.
pub(crate) async fn resolve_chunk_file_paths(
    kv_storage: &dyn edgequake_storage::traits::KVStorage,
    sources: &mut [SourceRef],
) {
    let chunk_doc_ids: Vec<String> = sources
        .iter()
        .filter(|s| s.source_type == "chunk" && s.file_path.is_none())
        .filter_map(|s| s.document_id.clone())
        .collect();

    if chunk_doc_ids.is_empty() {
        return;
    }

    let doc_names = resolve_document_names(kv_storage, &chunk_doc_ids).await;

    for source in sources.iter_mut() {
        if source.source_type == "chunk" && source.file_path.is_none() {
            if let Some(ref doc_id) = source.document_id {
                source.file_path = doc_names.get(doc_id).cloned();
            }
        }
    }
}

/// Resolve `[[table:<id>]]` pointers left in retrieved prose chunks (written by
/// `edgequake_pdf::mark_table_placeholders` on the chunker input) into the
/// table's GFM, **deduplicated by `table_id`**.
///
/// A table is rendered exactly once per response:
///   - if it's already returned as its own retrieved `kind="table"` chunk, the
///     prose pointer is stripped (the grid shows in that chunk);
///   - otherwise the first prose chunk referencing it gets the GFM hydrated
///     inline (fetched from the `chunks` table) so a referenced table still
///     reaches context; later references are stripped.
///
/// Best-effort: on any DB error the affected pointer is left/stripped without
/// failing the query. No-op when there's no Postgres pool.
#[cfg(feature = "postgres")]
pub(crate) async fn hydrate_table_references(
    pool: Option<&sqlx::PgPool>,
    sources: &mut [SourceRef],
) {
    use std::collections::{HashMap, HashSet};
    let Some(pool) = pool else { return };

    // table_ids already present as their own retrieved chunk (`{doc}-table-{id}`).
    let mut seen: HashSet<String> = HashSet::new();
    for s in sources.iter() {
        if s.kind.as_deref() == Some("table") {
            if let Some((_, tid)) = s.id.rsplit_once("-table-") {
                seen.insert(tid.to_string());
            }
        }
    }

    // Collect, per document, the referenced table ids not already present.
    let mut needed: HashMap<String, HashSet<String>> = HashMap::new();
    for s in sources.iter() {
        if s.kind.as_deref() == Some("table") {
            continue;
        }
        let (Some(doc), Some(snip)) = (s.document_id.as_deref(), s.snippet.as_deref()) else {
            continue;
        };
        for tid in edgequake_pdf::table_refs_in(snip) {
            if !seen.contains(&tid) {
                needed.entry(doc.to_string()).or_default().insert(tid);
            }
        }
    }

    // Fetch GFM for the needed (document, table) pairs.
    let mut gfm: HashMap<(String, String), String> = HashMap::new();
    for (doc, ids) in &needed {
        let Ok(doc_uuid) = uuid::Uuid::parse_str(doc) else {
            continue;
        };
        let id_list: Vec<String> = ids.iter().cloned().collect();
        let rows: Vec<(Option<String>, Option<String>, Option<serde_json::Value>)> =
            match sqlx::query_as(
                r#"SELECT table_id, content, table_rows
                     FROM chunks
                    WHERE document_id = $1 AND kind = 'table' AND table_id = ANY($2)"#,
            )
            .bind(doc_uuid)
            .bind(&id_list)
            .fetch_all(pool)
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(document_id = %doc, error = %e, "hydrate_table_references: fetch failed");
                    continue;
                }
            };
        for (tid, caption, table_rows) in rows {
            let Some(tid) = tid else { continue };
            if let Some(rendered) = crate::handlers::pdf_upload::content::render_table_markdown(
                caption.as_deref(),
                table_rows.as_ref(),
            ) {
                gfm.insert((doc.clone(), tid), rendered);
            }
        }
    }

    // Replace markers in prose snippets in rank order, deduping by table_id.
    for s in sources.iter_mut() {
        if s.kind.as_deref() == Some("table") {
            continue;
        }
        let Some(doc) = s.document_id.clone() else { continue };
        let Some(snip) = s.snippet.as_mut() else { continue };
        for tid in edgequake_pdf::table_refs_in(snip) {
            let marker = format!("[[table:{tid}]]");
            if !seen.contains(&tid) {
                if let Some(rendered) = gfm.get(&(doc.clone(), tid.clone())) {
                    *snip = snip.replace(&marker, &format!("\n\n{rendered}\n"));
                    seen.insert(tid);
                    continue;
                }
            }
            // Already present elsewhere, or not hydratable: drop the raw pointer
            // so `[[table:…]]` never leaks into the returned context.
            *snip = snip.replace(&marker, "");
        }
    }
}

/// Matches a `<div…>…</div>` caption block; group 1 is the inner text. Used to
/// locate figure-caption divs in retrieved prose chunks.
#[cfg(feature = "postgres")]
static CAPTION_DIV_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"(?s)<div[^>]*>(.*?)</div>").unwrap());

/// Normalize a caption for matching: collapse all whitespace runs to single
/// spaces, trim, lowercase. Both the div inner text and the stored figure
/// caption are normalized the same way so they compare equal despite OCR/markup
/// whitespace differences.
#[cfg(feature = "postgres")]
fn normalize_caption(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Inline figure-image markdown with a RELATIVE media URL. Consumers transform
/// it: the MCP tool fetches it as a base64 block, the OpenWebUI tool prepends
/// its public base. Relative (not absolute) because the backend doesn't know
/// the browser-facing base URL.
#[cfg(feature = "postgres")]
fn figure_image_md(caption: &str, document_id: &str, figure_id: &str) -> String {
    format!(
        "![{}](/api/v1/documents/{}/figures/{})",
        caption.trim(),
        document_id,
        figure_id
    )
}

/// Hydrate figure captions in retrieved chunks into inline image markdown so
/// every figure occurrence renders where it appears — not just figures whose
/// own `kind="figure"` chunk was retrieved.
///
/// Each occurrence is rewritten to
/// `![<caption>](/api/v1/documents/<doc>/figures/<figure_id>)`:
///   - a `kind="figure"` chunk's caption snippet (figure_id known directly);
///   - any `<div…>…</div>` in a text chunk whose inner text exact-normalized-
///     matches one of that document's figure captions (`chunks.content` where
///     `kind='figure'`).
///
/// **No dedup here** — a figure referenced N times yields N markdown tags;
/// consumers dedup as they see fit (the MCP tool by first occurrence, since
/// base64 is costly; OpenWebUI not at all, since URLs are cheap). Unmatched divs
/// are left untouched. Query-time only: embeddings and stored markdown are not
/// modified. No-op without a Postgres pool; best-effort on DB errors.
#[cfg(feature = "postgres")]
pub(crate) async fn hydrate_figure_captions(
    pool: Option<&sqlx::PgPool>,
    sources: &mut [SourceRef],
) {
    use std::collections::{HashMap, HashSet};
    let Some(pool) = pool else { return };

    // Documents present among the chunk sources.
    let docs: HashSet<String> = sources
        .iter()
        .filter(|s| s.source_type == "chunk")
        .filter_map(|s| s.document_id.clone())
        .collect();
    if docs.is_empty() {
        return;
    }

    // Per-document normalized-caption → figure_id map (for prose-div matching).
    let mut by_doc: HashMap<String, HashMap<String, String>> = HashMap::new();
    for doc in &docs {
        let Ok(doc_uuid) = uuid::Uuid::parse_str(doc) else {
            continue;
        };
        let rows: Vec<(Option<String>, Option<String>)> = match sqlx::query_as(
            r#"SELECT figure_id, content
                 FROM chunks
                WHERE document_id = $1 AND kind = 'figure' AND figure_id IS NOT NULL"#,
        )
        .bind(doc_uuid)
        .fetch_all(pool)
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(document_id = %doc, error = %e, "hydrate_figure_captions: fetch failed");
                continue;
            }
        };
        let mut map = HashMap::new();
        for (fid, caption) in rows {
            if let (Some(fid), Some(caption)) = (fid, caption) {
                let key = normalize_caption(&caption);
                if !key.is_empty() {
                    map.insert(key, fid);
                }
            }
        }
        if !map.is_empty() {
            by_doc.insert(doc.clone(), map);
        }
    }

    for s in sources.iter_mut() {
        if s.source_type != "chunk" {
            continue;
        }
        let Some(doc) = s.document_id.clone() else { continue };
        let Some(snip) = s.snippet.as_mut() else { continue };

        // Figure chunk: its snippet IS the caption; replace it wholesale (the
        // figure_id is known, no caption match needed).
        if s.kind.as_deref() == Some("figure") {
            if let Some(fid) = s.figure_id.clone() {
                let caption = snip.trim().to_string();
                *snip = figure_image_md(&caption, &doc, &fid);
            }
            continue;
        }

        // Text chunk: replace each caption div whose inner text matches a figure
        // caption for this document; leave non-matching divs as-is.
        let Some(map) = by_doc.get(&doc) else { continue };
        let replaced = CAPTION_DIV_RE.replace_all(snip, |caps: &regex::Captures| {
            let inner = &caps[1];
            match map.get(&normalize_caption(inner)) {
                Some(fid) => figure_image_md(inner, &doc, fid),
                None => caps[0].to_string(),
            }
        });
        *snip = replaced.into_owned();
    }
}

// Re-export workspace resolve functions for other modules
pub use workspace_resolve::{get_workspace_embedding_provider, get_workspace_vector_storage};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::TenantContext;
    use crate::state::AppState;
    use axum::extract::State;
    use axum::Json;

    #[tokio::test]
    async fn test_query_validation() {
        let state = AppState::test_state();
        let tenant_ctx = TenantContext::default();

        let request = QueryRequest {
            query: "".to_string(),
            mode: None,
            context_only: false,
            prompt_only: false,
            include_references: false,
            max_results: None,
            conversation_history: None,
            llm_provider: None,
            llm_model: None,
            system_prompt: None,
            document_filter: None,
        };

        let result = execute_query(State(state), tenant_ctx, Json(request)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_query_success() {
        let state = AppState::test_state();
        let tenant_ctx = TenantContext::default();

        let request = QueryRequest {
            query: "What is Rust?".to_string(),
            mode: Some("naive".to_string()),
            context_only: false,
            prompt_only: false,
            include_references: true,
            max_results: Some(5),
            conversation_history: None,
            llm_provider: None,
            llm_model: None,
            system_prompt: None,
            document_filter: None,
        };

        let result = execute_query(State(state), tenant_ctx, Json(request)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_stream_query_success() {
        let state = AppState::test_state();
        let tenant_ctx = TenantContext::default();

        let request = StreamQueryRequest {
            query: "What is Rust?".to_string(),
            mode: Some("naive".to_string()),
            system_prompt: None,
            document_filter: None,
            llm_provider: None,
            llm_model: None,
            stream_format: None,
        };

        let result = stream_query(State(state), tenant_ctx, Json(request)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_query_modes() {
        let state = AppState::test_state();
        let modes = vec!["naive", "local", "global", "hybrid", "mix"];

        for mode in modes {
            let tenant_ctx = TenantContext::default();
            let request = QueryRequest {
                query: "Test query".to_string(),
                mode: Some(mode.to_string()),
                context_only: false,
                prompt_only: false,
                include_references: false,
                max_results: None,
                conversation_history: None,
                llm_provider: None,
                llm_model: None,
                system_prompt: None,
                document_filter: None,
            };

            let result = execute_query(State(state.clone()), tenant_ctx, Json(request)).await;
            assert!(result.is_ok(), "Mode '{}' should succeed", mode);
        }
    }

    #[tokio::test]
    async fn test_query_with_context_only() {
        let state = AppState::test_state();
        let tenant_ctx = TenantContext::default();

        let request = QueryRequest {
            query: "What is Rust?".to_string(),
            mode: Some("naive".to_string()),
            context_only: true,
            prompt_only: false,
            include_references: false,
            max_results: Some(3),
            conversation_history: None,
            llm_provider: None,
            llm_model: None,
            system_prompt: None,
            document_filter: None,
        };

        let result = execute_query(State(state), tenant_ctx, Json(request)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_query_whitespace_only_fails() {
        let state = AppState::test_state();
        let tenant_ctx = TenantContext::default();

        let request = QueryRequest {
            query: "   \t\n   ".to_string(),
            mode: None,
            context_only: false,
            prompt_only: false,
            include_references: false,
            max_results: None,
            conversation_history: None,
            llm_provider: None,
            llm_model: None,
            system_prompt: None,
            document_filter: None,
        };

        let result = execute_query(State(state), tenant_ctx, Json(request)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_stream_query_empty_fails() {
        let state = AppState::test_state();
        let tenant_ctx = TenantContext::default();

        let request = StreamQueryRequest {
            query: "".to_string(),
            mode: None,
            system_prompt: None,
            document_filter: None,
            llm_provider: None,
            llm_model: None,
            stream_format: None,
        };

        let result = stream_query(State(state), tenant_ctx, Json(request)).await;
        assert!(result.is_err());
    }
}
