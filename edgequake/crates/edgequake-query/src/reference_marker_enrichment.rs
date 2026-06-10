//! Post-retrieval enrichment: resolve inline citation markers in surviving
//! chunks to their parsed references and append them to the chunk text.
//!
//! Unlike [`reference_code_enrichment`] / [`approved_algorithms_enrichment`],
//! this step does **no** vector search — it is a deterministic join. For each
//! surviving chunk it scans the text for numbered markers (`[5]`,
//! `[2,5,7]`), looks up that document's references by number, and appends a
//! compact "References cited:" block to `chunk.content`. Embeddings are
//! already computed at this point, so this only changes the prompt text.
//!
//! Silent no-op when the engine was built without a [`ReferenceStorage`].
//!
//! [`reference_code_enrichment`]: crate::reference_code_enrichment
//! [`approved_algorithms_enrichment`]: crate::approved_algorithms_enrichment

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use regex::Regex;

use edgequake_storage::traits::{DocumentReference, ReferenceStorage};

use crate::context::QueryContext;
use crate::engine::QueryRequest;

/// Max references appended per chunk. Overflow is summarised as "(+K more)".
const DEFAULT_MAX_REFS_PER_CHUNK: usize = 10;
/// Each reference's `raw_text` is trimmed to this many chars in the prompt.
const DEFAULT_MAX_RAW_LEN: usize = 250;

/// Inline numbered citation markers: a bracket of digits/commas/whitespace
/// only (`[5]`, `[2,5,7]`, `[2, 5]`). Ranges (`[2-7]`) are excluded.
static MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(\d+(?:\s*,\s*\d+)*)\]").expect("valid marker regex"));

/// Scan text for inline citation markers, returning cited numbers in
/// first-seen order, de-duplicated.
fn scan_citation_markers(text: &str) -> Vec<i32> {
    let mut seen: Vec<i32> = Vec::new();
    for caps in MARKER_RE.captures_iter(text) {
        for part in caps.get(1).unwrap().as_str().split(',') {
            if let Ok(n) = part.trim().parse::<i32>() {
                if !seen.contains(&n) {
                    seen.push(n);
                }
            }
        }
    }
    seen
}

/// Enrich surviving chunks with the references their `[n]` markers cite.
/// Best-effort — errors are logged and swallowed so answer generation always
/// proceeds.
pub async fn enrich_with_references(
    context: &mut QueryContext,
    request: &QueryRequest,
    storage: Option<&Arc<dyn ReferenceStorage>>,
) {
    let Some(storage) = storage else {
        return; // Feature disabled.
    };
    if let Err(e) = try_enrich(context, request, storage).await {
        tracing::warn!(error = %e, "reference-marker enrichment skipped");
    }
}

async fn try_enrich(
    context: &mut QueryContext,
    request: &QueryRequest,
    storage: &Arc<dyn ReferenceStorage>,
) -> Result<(), EnrichmentError> {
    let tenant_id = request.tenant_id().ok_or(EnrichmentError::MissingTenant)?;
    let workspace_id = request
        .workspace_id()
        .ok_or(EnrichmentError::MissingWorkspace)?;
    let tenant_uuid = uuid::Uuid::parse_str(&tenant_id)
        .map_err(|e| EnrichmentError::Other(format!("invalid tenant_id: {e}")))?;
    let workspace_uuid = uuid::Uuid::parse_str(&workspace_id)
        .map_err(|e| EnrichmentError::Other(format!("invalid workspace_id: {e}")))?;

    let max_per_chunk: usize = std::env::var("EDGEQUAKE_QUERY_REFS_PER_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_REFS_PER_CHUNK);
    let max_raw_len: usize = std::env::var("EDGEQUAKE_QUERY_REF_RAW_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_RAW_LEN);

    let mut enriched_chunks = 0usize;
    for chunk in context.chunks.iter_mut() {
        let Some(document_id) = chunk.document_id.as_ref().filter(|s| !s.is_empty()) else {
            continue;
        };
        let cited = scan_citation_markers(&chunk.content);
        if cited.is_empty() {
            continue;
        }

        let refs = storage
            .references_by_numbers(tenant_uuid, workspace_uuid, document_id, &cited)
            .await
            .map_err(|e| EnrichmentError::Storage(e.to_string()))?;
        if refs.is_empty() {
            continue;
        }

        // Index resolved references by number, then emit in marker (first-seen)
        // order so the block reads in citation order, not numeric order.
        let by_number: HashMap<i32, &DocumentReference> =
            refs.iter().map(|r| (r.reference_number, r)).collect();
        let resolved: Vec<&DocumentReference> = cited
            .iter()
            .filter_map(|n| by_number.get(n).copied())
            .collect();
        if resolved.is_empty() {
            continue;
        }

        let block = render_block(&resolved, max_per_chunk, max_raw_len);
        chunk.content.push_str(&block);
        enriched_chunks += 1;
    }

    if enriched_chunks > 0 {
        tracing::info!(
            chunks = enriched_chunks,
            "appended resolved references to retrieved chunks"
        );
    }
    Ok(())
}

/// Render the "References cited:" block appended to a chunk.
fn render_block(resolved: &[&DocumentReference], max_per_chunk: usize, max_raw_len: usize) -> String {
    let shown = resolved.len().min(max_per_chunk);
    let mut block = String::from("\n\nReferences cited:");
    for r in &resolved[..shown] {
        block.push_str("\n[");
        block.push_str(&r.reference_number.to_string());
        block.push_str("] ");
        block.push_str(&truncate(&r.raw_text, max_raw_len));
    }
    let overflow = resolved.len().saturating_sub(shown);
    if overflow > 0 {
        block.push_str(&format!("\n(+{overflow} more)"));
    }
    block
}

/// Trim to at most `max` chars (char-boundary safe), appending an ellipsis
/// when truncated.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[derive(Debug, thiserror::Error)]
enum EnrichmentError {
    #[error("request missing tenant_id")]
    MissingTenant,
    #[error("request missing workspace_id")]
    MissingWorkspace,
    #[error("reference storage error: {0}")]
    Storage(String),
    #[error("{0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(num: i32, text: &str) -> DocumentReference {
        DocumentReference {
            id: uuid::Uuid::nil(),
            document_id: "doc-1".to_string(),
            reference_number: num,
            raw_text: text.to_string(),
            doi: None,
            url: None,
        }
    }

    #[test]
    fn scans_markers_first_seen_order() {
        assert_eq!(
            scan_citation_markers("see [5], then [2, 5] and [7]"),
            vec![5, 2, 7]
        );
    }

    #[test]
    fn renders_in_marker_order_with_overflow() {
        let refs = vec![r(2, "Second"), r(5, "Fifth"), r(7, "Seventh")];
        let resolved: Vec<&DocumentReference> = refs.iter().collect();
        let block = render_block(&resolved, 2, 250);
        assert!(block.starts_with("\n\nReferences cited:"));
        assert!(block.contains("[2] Second"));
        assert!(block.contains("[5] Fifth"));
        assert!(!block.contains("Seventh"));
        assert!(block.contains("(+1 more)"));
    }

    #[test]
    fn truncates_long_raw_text() {
        let long = "x".repeat(300);
        let t = truncate(&long, 250);
        assert_eq!(t.chars().count(), 250);
        assert!(t.ends_with('…'));
    }
}
