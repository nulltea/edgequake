//! Postgres implementation of reference-codebase vector search.

use async_trait::async_trait;
use sqlx::{postgres::PgRow as RawPgRow, FromRow, PgPool, Row as SqlxRow};
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{Result, StorageError};
use crate::traits::{ReferenceCodebaseSearchHit, ReferenceCodebaseVectorStorage};

pub struct PgReferenceCodebaseVectorStorage {
    pool: Arc<PgPool>,
}

impl PgReferenceCodebaseVectorStorage {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool: Arc::new(pool),
        }
    }
}

#[async_trait]
impl ReferenceCodebaseVectorStorage for PgReferenceCodebaseVectorStorage {
    async fn search_reference_codebase(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        query_vec: &[f32],
        limit: i64,
        max_distance: f64,
        document_repo_id: Option<Uuid>,
        index_id: Option<Uuid>,
        algorithm_ids: Option<&[Uuid]>,
    ) -> Result<Vec<ReferenceCodebaseSearchHit>> {
        let literal = vector_literal(query_vec);
        let algorithm_ids_vec = algorithm_ids.map(|ids| ids.to_vec()).unwrap_or_default();
        let use_algorithm_filter = !algorithm_ids_vec.is_empty();

        let rows = sqlx::query_as::<_, Row>(SQL)
            .bind(&literal)
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(limit)
            .bind(max_distance)
            .bind(document_repo_id)
            .bind(index_id)
            .bind(&algorithm_ids_vec)
            .bind(use_algorithm_filter)
            .fetch_all(&*self.pool)
            .await
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(rows.into_iter().map(Row::into_hit).collect())
    }

    async fn fetch_chunks_by_symbol_names(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        names: &[String],
        document_repo_id: Option<Uuid>,
        index_id: Option<Uuid>,
        limit: i64,
    ) -> Result<Vec<ReferenceCodebaseSearchHit>> {
        if names.is_empty() {
            return Ok(Vec::new());
        }
        let rows = sqlx::query_as::<_, Row>(ENTITY_SQL)
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(names)
            .bind(document_repo_id)
            .bind(index_id)
            .bind(limit)
            .fetch_all(&*self.pool)
            .await
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| {
                let matched = r.symbol_name.clone();
                let mut hit = r.into_hit();
                hit.matched_entity = matched;
                hit
            })
            .collect())
    }
}

// Match chunks three ways:
//   (a) symbol_name = name                 — exact match on stored qualified form.
//   (b) symbol_name LIKE name || '::%'     — query said `SoftmaxEvaluator`; stored
//                                            form is `SoftmaxEvaluator::softmax`.
//   (c) symbol_name LIKE '%::' || name     — query said `encrypt`; stored form is
//                                            `seal::Encryptor::encrypt`.
// Each candidate in $3 is probed against all three; chunks.symbol_name is still
// an exact key per row so the index on it covers (a).
const ENTITY_SQL: &str = r#"
    SELECT
        c.id                    AS chunk_id,
        c.index_id              AS index_id,
        c.document_id           AS document_id,
        c.document_repo_id      AS document_repo_id,
        i.repo_url              AS repo_url,
        i.repo_commit           AS repo_commit,
        c.file_path             AS file_path,
        c.language              AS language,
        c.symbol_name           AS symbol_name,
        c.start_line            AS start_line,
        c.end_line              AS end_line,
        c.chunk_kind            AS chunk_kind,
        c.algorithm_id          AS algorithm_id,
        c.content               AS content,
        0.0::float8             AS distance
    FROM reference_codebase_chunks c
    JOIN reference_codebase_indexes i ON i.id = c.index_id
    WHERE c.tenant_id = $1
      AND c.workspace_id = $2
      AND i.status = 'complete'
      AND ($4::uuid IS NULL OR c.document_repo_id = $4)
      AND ($5::uuid IS NULL OR c.index_id = $5)
      AND (
           c.symbol_name = ANY($3::text[])
        OR EXISTS (
               SELECT 1 FROM unnest($3::text[]) AS pat
                WHERE c.symbol_name LIKE pat || '::%'
                   OR c.symbol_name LIKE '%::' || pat
           )
      )
    ORDER BY c.symbol_name, c.start_line
    LIMIT $6
"#;

const SQL: &str = r#"
    WITH matches AS (
        SELECT
            rce.chunk_id,
            rce.index_id,
            rce.embedding <=> $1::vector AS distance
        FROM reference_codebase_embeddings rce
        JOIN reference_codebase_indexes rci ON rci.id = rce.index_id
        WHERE rce.tenant_id = $2
          AND rce.workspace_id = $3
          AND rci.status = 'complete'
          AND ($6::uuid IS NULL OR rce.document_repo_id = $6)
          AND ($7::uuid IS NULL OR rce.index_id = $7)
          AND ($9::bool = false OR rce.algorithm_id = ANY($8::uuid[]))
        ORDER BY rce.embedding <=> $1::vector ASC
        LIMIT $4
    )
    SELECT
        c.id                    AS chunk_id,
        c.index_id              AS index_id,
        c.document_id           AS document_id,
        c.document_repo_id      AS document_repo_id,
        i.repo_url              AS repo_url,
        i.repo_commit           AS repo_commit,
        c.file_path             AS file_path,
        c.language              AS language,
        c.symbol_name           AS symbol_name,
        c.start_line            AS start_line,
        c.end_line              AS end_line,
        c.chunk_kind            AS chunk_kind,
        c.algorithm_id          AS algorithm_id,
        c.content               AS content,
        m.distance::float8      AS distance
    FROM matches m
    JOIN reference_codebase_chunks c ON c.id = m.chunk_id
    JOIN reference_codebase_indexes i ON i.id = c.index_id
    WHERE m.distance <= $5
    ORDER BY
        (c.algorithm_focus * -0.08 + m.distance) ASC,
        m.distance ASC
"#;

struct Row {
    chunk_id: Uuid,
    index_id: Uuid,
    document_id: String,
    document_repo_id: Uuid,
    repo_url: String,
    repo_commit: String,
    file_path: String,
    language: String,
    symbol_name: Option<String>,
    start_line: i32,
    end_line: i32,
    chunk_kind: String,
    algorithm_id: Option<Uuid>,
    content: String,
    distance: f64,
}

impl<'r> FromRow<'r, RawPgRow> for Row {
    fn from_row(row: &'r RawPgRow) -> std::result::Result<Self, sqlx::Error> {
        Ok(Self {
            chunk_id: row.try_get("chunk_id")?,
            index_id: row.try_get("index_id")?,
            document_id: row.try_get("document_id")?,
            document_repo_id: row.try_get("document_repo_id")?,
            repo_url: row.try_get("repo_url")?,
            repo_commit: row.try_get("repo_commit")?,
            file_path: row.try_get("file_path")?,
            language: row.try_get("language")?,
            symbol_name: row.try_get("symbol_name")?,
            start_line: row.try_get("start_line")?,
            end_line: row.try_get("end_line")?,
            chunk_kind: row.try_get("chunk_kind")?,
            algorithm_id: row.try_get("algorithm_id")?,
            content: row.try_get("content")?,
            distance: row.try_get("distance")?,
        })
    }
}

impl Row {
    fn into_hit(self) -> ReferenceCodebaseSearchHit {
        ReferenceCodebaseSearchHit {
            chunk_id: self.chunk_id,
            index_id: self.index_id,
            document_id: self.document_id,
            document_repo_id: self.document_repo_id,
            repo_url: self.repo_url,
            repo_commit: self.repo_commit,
            file_path: self.file_path,
            language: self.language,
            symbol_name: self.symbol_name,
            start_line: self.start_line,
            end_line: self.end_line,
            chunk_kind: self.chunk_kind,
            algorithm_id: self.algorithm_id,
            content: self.content,
            cosine_distance: self.distance,
            matched_entity: None,
            bm25_score: None,
            final_score: None,
        }
    }
}

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
