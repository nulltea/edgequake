//! Post-retrieval enrichment: attach approved reference-code snippets to a
//! query context (Phase 1 of the Reference Code GraphRAG extension).
//!
//! Called by the query engine after `balance_context`. Joins the
//! `code_artifacts` + `algorithms` tables, keyed on the document-ids that
//! appeared in retrieved chunks, and attaches the approved snippets to
//! [`QueryContext::reference_code`]. [`context::to_context_string`] renders
//! them as a `### Reference Code Implementations` section for the LLM.
//!
//! **Silently skipped** when:
//! - The `postgres` feature is disabled (compile-time skip),
//! - `DATABASE_URL` is unset (test / in-memory runs),
//! - the request has no tenant/workspace (non-multi-tenant legacy path),
//! - the retrieval produced no chunks (nothing to enrich).

use crate::context::QueryContext;
#[cfg(feature = "postgres")]
use crate::context::ReferenceCodeSnippet;
use crate::engine::QueryRequest;

/// Max approved snippets we'll inline. Kept conservative by default; the
/// snippets are bounded to ~8 KB each by the analyzer's snippet extractor
/// already, so 5 × 8 KB = 40 KB is a safe upper bound for most models.
#[cfg(feature = "postgres")]
const DEFAULT_MAX_SNIPPETS: usize = 5;

/// Enrich `context` with approved reference-code snippets for the documents
/// that appear in its chunks. No-op on failure; logs a warning so downstream
/// answer generation always proceeds with whatever context was built.
pub async fn enrich_with_reference_code(context: &mut QueryContext, request: &QueryRequest) {
    #[cfg(feature = "postgres")]
    {
        if let Err(e) = try_enrich(context, request).await {
            tracing::warn!(error = %e, "reference-code enrichment skipped");
        }
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (context, request);
    }
}

#[cfg(feature = "postgres")]
async fn try_enrich(
    context: &mut QueryContext,
    request: &QueryRequest,
) -> Result<(), EnrichmentError> {
    // Collect distinct document_ids from retrieval. No docs → nothing to enrich.
    let mut document_ids: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for chunk in &context.chunks {
        if let Some(ref id) = chunk.document_id {
            if !id.is_empty() {
                document_ids.insert(id.clone());
            }
        }
    }
    if document_ids.is_empty() {
        return Ok(());
    }

    let tenant_id = request.tenant_id().ok_or(EnrichmentError::MissingTenant)?;
    let workspace_id = request
        .workspace_id()
        .ok_or(EnrichmentError::MissingWorkspace)?;
    let tenant_uuid = uuid::Uuid::parse_str(&tenant_id)
        .map_err(|e| EnrichmentError::Other(format!("invalid tenant_id: {e}")))?;
    let workspace_uuid = uuid::Uuid::parse_str(&workspace_id)
        .map_err(|e| EnrichmentError::Other(format!("invalid workspace_id: {e}")))?;

    let database_url =
        std::env::var("DATABASE_URL").map_err(|_| EnrichmentError::DatabaseUrlUnset)?;
    let pool = sqlx::PgPool::connect(&database_url)
        .await
        .map_err(|e| EnrichmentError::Db(e.to_string()))?;

    let docs: Vec<String> = document_ids.into_iter().collect();
    let max_snippets: i64 = std::env::var("EDGEQUAKE_QUERY_CODE_SNIPPETS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_SNIPPETS as i64);

    // Join algorithms + code_artifacts + document_repos (for repo_url).
    // Multi-tenant isolation enforced via tenant_id + workspace_id WHERE.
    let rows: Vec<PgRow> = sqlx::query_as::<_, PgRow>(
        r#"
        SELECT
            ca.algorithm_id::text      AS algorithm_id,
            a.name                     AS algorithm_name,
            ca.document_id             AS document_id,
            ca.file_path               AS file_path,
            ca.start_line              AS start_line,
            ca.end_line                AS end_line,
            ca.language                AS language,
            ca.snippet                 AS snippet,
            ca.repo_commit             AS repo_commit,
            ca.match_rationale         AS match_rationale,
            dr.url                     AS repo_url,
            CASE ca.match_confidence WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END AS conf_rank
        FROM code_artifacts ca
        JOIN algorithms a ON a.id = ca.algorithm_id
        LEFT JOIN document_repos dr ON dr.id = ca.document_repo_id
        WHERE ca.tenant_id = $1
          AND ca.workspace_id = $2
          AND ca.document_id = ANY($3)
          AND ca.status = 'approved'
        ORDER BY conf_rank ASC, ca.created_at ASC
        LIMIT $4
        "#,
    )
    .bind(tenant_uuid)
    .bind(workspace_uuid)
    .bind(&docs)
    .bind(max_snippets)
    .fetch_all(&pool)
    .await
    .map_err(|e| EnrichmentError::Db(e.to_string()))?;

    if rows.is_empty() {
        tracing::debug!(
            doc_count = docs.len(),
            "no approved code_artifacts for retrieval documents"
        );
        return Ok(());
    }

    context.reference_code.reserve(rows.len());
    for r in rows {
        context.reference_code.push(ReferenceCodeSnippet {
            algorithm_id: r.algorithm_id,
            algorithm_name: r.algorithm_name,
            document_id: r.document_id,
            file_path: r.file_path,
            start_line: r.start_line,
            end_line: r.end_line,
            language: r.language,
            snippet: r.snippet,
            repo_url: r.repo_url,
            repo_commit: r.repo_commit,
            match_rationale: r.match_rationale,
        });
    }
    tracing::info!(
        snippets = context.reference_code.len(),
        doc_count = docs.len(),
        "attached approved reference-code snippets to context"
    );
    Ok(())
}

#[cfg(feature = "postgres")]
#[derive(Debug, thiserror::Error)]
enum EnrichmentError {
    #[error("DATABASE_URL not set — skipping reference-code enrichment")]
    DatabaseUrlUnset,
    #[error("request missing tenant_id — skipping reference-code enrichment")]
    MissingTenant,
    #[error("request missing workspace_id — skipping reference-code enrichment")]
    MissingWorkspace,
    #[error("db error during reference-code enrichment: {0}")]
    Db(String),
    #[error("{0}")]
    Other(String),
}

#[cfg(feature = "postgres")]
#[derive(sqlx::FromRow)]
struct PgRow {
    algorithm_id: String,
    algorithm_name: String,
    document_id: String,
    file_path: String,
    start_line: i32,
    end_line: i32,
    language: String,
    snippet: String,
    repo_commit: String,
    match_rationale: Option<String>,
    repo_url: Option<String>,
    #[allow(dead_code)]
    conf_rank: i32,
}
