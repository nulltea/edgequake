//! Backfill parsed references for every document in the default workspace.
//!
//! The inline reference-parsing step only runs during ingestion, so documents
//! that were indexed before the feature existed have no references. This tool
//! reads each document's stored markdown (`documents.content`), runs the
//! **same** deterministic parser (`edgequake_pipeline::parse_references`), and
//! replaces its rows in `document_references`.
//!
//! It does **not** reprocess documents — no chunking, embedding, or LLM work.
//! It is idempotent: re-running overwrites each document's references.
//!
//! Note: documents whose `content` was truncated at ingestion (some early
//! text-upload paths stored only a 500-char summary) won't contain a
//! reference section and are reported as `no_content`/skipped — they need a
//! real reprocess to recover full text.
//!
//! Usage:
//!   DATABASE_URL=postgres://user:pass@host:5432/db \
//!     cargo run --example backfill_references --features postgres
//!
//! Environment:
//!   DATABASE_URL     (required) Postgres connection string.
//!   EQ_WORKSPACE_ID  Workspace UUID. Default: the built-in default workspace
//!                    (00000000-0000-0000-0000-000000000003).
//!   EQ_DRY_RUN       When set (any value), parse and report counts but do not
//!                    write any rows.

use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use uuid::Uuid;

use edgequake_pipeline::parse_references;
use edgequake_storage::traits::{NewDocumentReference, ReferenceStorage};
use edgequake_storage::PgReferenceStorage;

/// Built-in default workspace / tenant (see `workspace_service_impl.rs`).
const DEFAULT_WORKSPACE_ID: &str = "00000000-0000-0000-0000-000000000003";
const DEFAULT_TENANT_ID: &str = "00000000-0000-0000-0000-000000000002";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let database_url = std::env::var("DATABASE_URL")
        .map_err(|_| anyhow::anyhow!("DATABASE_URL is required"))?;
    let workspace_id = Uuid::parse_str(
        &std::env::var("EQ_WORKSPACE_ID").unwrap_or_else(|_| DEFAULT_WORKSPACE_ID.to_string()),
    )?;
    let default_tenant = Uuid::parse_str(DEFAULT_TENANT_ID)?;
    let dry_run = std::env::var("EQ_DRY_RUN").is_ok();

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    println!("Backfilling references — workspace={workspace_id} dry_run={dry_run}");

    // Enumerate documents for the workspace, pulling the FULL markdown.
    //
    // `documents.content` is truncated to 64KB for PDF-origin documents
    // (see pdf_processing.rs), which cuts the reference section off any paper
    // longer than 64KB. The untruncated markdown lives in
    // `pdf_documents.markdown_content`, so prefer that and fall back to
    // `documents.content` for non-PDF documents. tenant_id is written
    // alongside so backfilled rows match the scope the query/enrichment paths
    // read with.
    let doc_rows = sqlx::query(
        "SELECT d.id::text AS id, d.tenant_id::text AS tenant_id, \
                COALESCE(p.markdown_content, d.content) AS content \
         FROM documents d \
         LEFT JOIN pdf_documents p ON p.document_id = d.id \
         WHERE d.workspace_id = $1 ORDER BY d.created_at",
    )
    .bind(workspace_id)
    .fetch_all(&pool)
    .await?;

    let storage = PgReferenceStorage::new(pool.clone());

    let mut total_docs = 0usize;
    let mut docs_with_refs = 0usize;
    let mut total_refs = 0usize;
    let mut no_content = 0usize;

    for row in &doc_rows {
        total_docs += 1;
        let document_id: String = row.get("id");
        let tenant_id: Uuid = row
            .get::<Option<String>, _>("tenant_id")
            .and_then(|s| Uuid::parse_str(&s).ok())
            .unwrap_or(default_tenant);

        let content: Option<String> = row.get("content");
        let Some(content) = content.filter(|c| !c.trim().is_empty()) else {
            no_content += 1;
            continue;
        };

        let parsed = parse_references(&content);
        if parsed.is_empty() {
            continue;
        }
        docs_with_refs += 1;
        total_refs += parsed.len();

        if dry_run {
            println!("[dry-run] {document_id}: {} references", parsed.len());
            continue;
        }

        let rows: Vec<NewDocumentReference> = parsed
            .into_iter()
            .map(|r| NewDocumentReference {
                reference_number: r.number as i32,
                raw_text: r.raw_text,
                doi: r.doi,
                url: r.url,
            })
            .collect();
        let n = rows.len();
        storage
            .replace_references(tenant_id, workspace_id, &document_id, &rows)
            .await?;
        println!("{document_id}: wrote {n} references");
    }

    println!(
        "\nDone. documents={total_docs} with_references={docs_with_refs} \
         total_references={total_refs} no_content={no_content} dry_run={dry_run}"
    );
    Ok(())
}
