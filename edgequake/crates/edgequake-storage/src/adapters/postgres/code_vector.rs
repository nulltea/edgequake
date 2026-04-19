//! Postgres implementation of [`CodeVectorStorage`].
//!
//! Runs an HNSW cosine search over `code_artifact_embeddings`, joined to
//! `code_artifacts` + `algorithms` + `document_repos` for the rendering
//! fields. Caller-facing rows are [`CodeSearchHit`]s — no SQL leaks.

use async_trait::async_trait;
use sqlx::{postgres::PgRow as RawPgRow, FromRow, PgPool, Row as SqlxRow};
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{Result, StorageError};
use crate::traits::{CodeSearchHit, CodeVectorStorage};

pub struct PgCodeVectorStorage {
    pool: Arc<PgPool>,
}

impl PgCodeVectorStorage {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool: Arc::new(pool),
        }
    }
}

#[async_trait]
impl CodeVectorStorage for PgCodeVectorStorage {
    async fn search_approved_code(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        query_vec: &[f32],
        limit: i64,
        max_distance: f64,
        document_ids: Option<&[String]>,
    ) -> Result<Vec<CodeSearchHit>> {
        let literal = vector_literal(query_vec);

        // When document_ids is Some(non-empty), restrict the vector search
        // to those documents so we don't surface code from papers that
        // main retrieval didn't hit. The filter is pushed *into* the CTE
        // so pgvector's HNSW index applies after the pre-filter — scoring
        // is still top-k by distance, but only among allowed documents.
        let use_filter = matches!(document_ids, Some(ids) if !ids.is_empty());

        let rows: Vec<Row> = if use_filter {
            let ids: &[String] = document_ids.unwrap();
            sqlx::query_as::<_, Row>(SQL_FILTERED)
                .bind(&literal)
                .bind(tenant_id)
                .bind(workspace_id)
                .bind(limit)
                .bind(max_distance)
                .bind(ids)
                .fetch_all(&*self.pool)
                .await
        } else {
            sqlx::query_as::<_, Row>(SQL)
                .bind(&literal)
                .bind(tenant_id)
                .bind(workspace_id)
                .bind(limit)
                .bind(max_distance)
                .fetch_all(&*self.pool)
                .await
        }
        .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(rows.into_iter().map(Row::into_hit).collect())
    }
}

const SQL: &str = r#"
    WITH matches AS (
        SELECT
            cae.code_artifact_id,
            cae.embedding <=> $1::vector AS distance
        FROM code_artifact_embeddings cae
        WHERE cae.tenant_id = $2
          AND cae.workspace_id = $3
        ORDER BY cae.embedding <=> $1::vector ASC
        LIMIT $4
    )
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
        m.distance::float8         AS distance
    FROM matches m
    JOIN code_artifacts ca ON ca.id = m.code_artifact_id
    JOIN algorithms a ON a.id = ca.algorithm_id
    LEFT JOIN document_repos dr ON dr.id = ca.document_repo_id
    WHERE ca.status = 'approved'
      AND m.distance <= $5
    ORDER BY m.distance ASC
"#;

/// Same as [`SQL`] but pre-filters to `code_artifacts.document_id = ANY($6)`
/// inside the CTE. This keeps the vector-search top-K honest — without
/// pushing the filter into the CTE, a paper-unrelated snippet could evict
/// the right one from the top-K before the WHERE runs.
const SQL_FILTERED: &str = r#"
    WITH matches AS (
        SELECT
            cae.code_artifact_id,
            cae.embedding <=> $1::vector AS distance
        FROM code_artifact_embeddings cae
        JOIN code_artifacts ca2 ON ca2.id = cae.code_artifact_id
        WHERE cae.tenant_id = $2
          AND cae.workspace_id = $3
          AND ca2.document_id = ANY($6)
        ORDER BY cae.embedding <=> $1::vector ASC
        LIMIT $4
    )
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
        m.distance::float8         AS distance
    FROM matches m
    JOIN code_artifacts ca ON ca.id = m.code_artifact_id
    JOIN algorithms a ON a.id = ca.algorithm_id
    LEFT JOIN document_repos dr ON dr.id = ca.document_repo_id
    WHERE ca.status = 'approved'
      AND m.distance <= $5
    ORDER BY m.distance ASC
"#;

struct Row {
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
    distance: f64,
}

impl<'r> FromRow<'r, RawPgRow> for Row {
    fn from_row(row: &'r RawPgRow) -> std::result::Result<Self, sqlx::Error> {
        Ok(Self {
            algorithm_id: row.try_get("algorithm_id")?,
            algorithm_name: row.try_get("algorithm_name")?,
            document_id: row.try_get("document_id")?,
            file_path: row.try_get("file_path")?,
            start_line: row.try_get("start_line")?,
            end_line: row.try_get("end_line")?,
            language: row.try_get("language")?,
            snippet: row.try_get("snippet")?,
            repo_commit: row.try_get("repo_commit")?,
            match_rationale: row.try_get("match_rationale")?,
            repo_url: row.try_get("repo_url")?,
            distance: row.try_get("distance")?,
        })
    }
}

impl Row {
    fn into_hit(self) -> CodeSearchHit {
        CodeSearchHit {
            algorithm_id: self.algorithm_id,
            algorithm_name: self.algorithm_name,
            document_id: self.document_id,
            file_path: self.file_path,
            start_line: self.start_line,
            end_line: self.end_line,
            language: self.language,
            snippet: self.snippet,
            repo_url: self.repo_url,
            repo_commit: self.repo_commit,
            match_rationale: self.match_rationale,
            cosine_distance: self.distance,
        }
    }
}

/// Serialize a slice to the pgvector literal form `[v1,v2,...]`.
fn vector_literal(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 10 + 2);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        use std::fmt::Write;
        let _ = write!(s, "{:.7}", x);
    }
    s.push(']');
    s
}
