//! Storage for `code_artifacts` + `code_reference_runs`.
//!
//! Same pattern as `crate::repo_detection::storage`: trait + Postgres impl
//! gated on the `postgres` feature, runtime-checked sqlx queries.

use async_trait::async_trait;
use uuid::Uuid;

use super::types::{
    ArtifactStatus, CodeArtifact, CodeArtifactCandidate, CodeReferenceRun, RunStatus,
};

#[derive(Debug, thiserror::Error)]
pub enum CodeStorageError {
    #[cfg(feature = "postgres")]
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("decode error: {0}")]
    Decode(String),
}

#[async_trait]
pub trait CodeArtifactStorage: Send + Sync {
    /// Persist candidates idempotently via the natural-key upsert (see
    /// migration 043). Existing rows keep their review status.
    async fn upsert_candidates(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        candidates: &[CodeArtifactCandidate],
    ) -> Result<Vec<CodeArtifact>, CodeStorageError>;

    async fn list_for_document(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
    ) -> Result<Vec<CodeArtifact>, CodeStorageError>;

    async fn list_for_algorithm(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        algorithm_id: Uuid,
    ) -> Result<Vec<CodeArtifact>, CodeStorageError>;

    async fn update_status(
        &self,
        id: Uuid,
        tenant_id: Uuid,
        workspace_id: Uuid,
        status: ArtifactStatus,
    ) -> Result<bool, CodeStorageError>;

    async fn upsert_run(&self, run: &CodeReferenceRun) -> Result<(), CodeStorageError>;

    async fn mark_run_status(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        document_repo_id: Uuid,
        status: RunStatus,
    ) -> Result<(), CodeStorageError>;

    #[allow(clippy::too_many_arguments)]
    async fn mark_run_complete(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        document_repo_id: Uuid,
        algorithm_count: i32,
        finding_count: i32,
        cost_usd_equivalent: Option<f32>,
    ) -> Result<(), CodeStorageError>;

    async fn mark_run_failed(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        document_repo_id: Uuid,
        error_message: &str,
    ) -> Result<(), CodeStorageError>;

    async fn get_run(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        document_repo_id: Uuid,
    ) -> Result<Option<CodeReferenceRun>, CodeStorageError>;

    /// Count code artifacts grouped by document for a workspace. Drives
    /// the document-list UI's per-row "Code matches" action button
    /// visibility (like `AlgorithmStorage::counts_by_document`).
    async fn counts_by_document(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
    ) -> Result<Vec<(String, i64)>, CodeStorageError>;
}

#[cfg(feature = "postgres")]
pub use postgres::PostgresCodeArtifactStorage;

#[cfg(feature = "postgres")]
mod postgres {
    use super::*;
    use chrono::{DateTime, Utc};
    use sqlx::{postgres::PgRow, PgPool, Row};

    use crate::code_analysis::types::MatchConfidence;

    pub struct PostgresCodeArtifactStorage {
        pool: PgPool,
    }

    impl PostgresCodeArtifactStorage {
        pub fn new(pool: PgPool) -> Self {
            Self { pool }
        }
    }

    fn decode_artifact(row: PgRow) -> Result<CodeArtifact, CodeStorageError> {
        let status_s: String = row.try_get("status")?;
        let confidence_s: String = row.try_get("match_confidence")?;
        let status = ArtifactStatus::parse(&status_s)
            .ok_or_else(|| CodeStorageError::Decode(format!("bad status: {status_s}")))?;
        let match_confidence = MatchConfidence::parse(&confidence_s)
            .ok_or_else(|| CodeStorageError::Decode(format!("bad confidence: {confidence_s}")))?;
        Ok(CodeArtifact {
            id: row.try_get("id")?,
            tenant_id: row.try_get("tenant_id")?,
            workspace_id: row.try_get("workspace_id")?,
            document_id: row.try_get("document_id")?,
            algorithm_id: row.try_get("algorithm_id")?,
            document_repo_id: row.try_get("document_repo_id")?,
            repo_commit: row.try_get("repo_commit")?,
            repo_license: row.try_get("repo_license")?,
            language: row.try_get("language")?,
            file_path: row.try_get("file_path")?,
            symbol_name: row.try_get("symbol_name")?,
            start_line: row.try_get("start_line")?,
            end_line: row.try_get("end_line")?,
            snippet: row.try_get("snippet")?,
            match_rationale: row.try_get("match_rationale")?,
            match_confidence,
            status,
            embedding_id: row.try_get("embedding_id")?,
            created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
            updated_at: row.try_get::<DateTime<Utc>, _>("updated_at")?,
        })
    }

    fn decode_run(row: PgRow) -> Result<CodeReferenceRun, CodeStorageError> {
        let status_s: String = row.try_get("status")?;
        let status = RunStatus::parse(&status_s)
            .ok_or_else(|| CodeStorageError::Decode(format!("bad run status: {status_s}")))?;
        Ok(CodeReferenceRun {
            tenant_id: row.try_get("tenant_id")?,
            workspace_id: row.try_get("workspace_id")?,
            document_id: row.try_get("document_id")?,
            document_repo_id: row.try_get("document_repo_id")?,
            status,
            algorithm_count: row.try_get("algorithm_count")?,
            finding_count: row.try_get("finding_count")?,
            cost_usd_equivalent: row.try_get("cost_usd_equivalent")?,
            error_message: row.try_get("error_message")?,
            attempted_at: row.try_get::<DateTime<Utc>, _>("attempted_at")?,
            completed_at: row.try_get::<Option<DateTime<Utc>>, _>("completed_at")?,
        })
    }

    #[async_trait]
    impl CodeArtifactStorage for PostgresCodeArtifactStorage {
        async fn upsert_candidates(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            candidates: &[CodeArtifactCandidate],
        ) -> Result<Vec<CodeArtifact>, CodeStorageError> {
            if candidates.is_empty() {
                return Ok(Vec::new());
            }
            let mut tx = self.pool.begin().await?;
            let mut out = Vec::with_capacity(candidates.len());
            for c in candidates {
                let row = sqlx::query(
                    r#"
                    INSERT INTO code_artifacts (
                        id, tenant_id, workspace_id, document_id,
                        algorithm_id, document_repo_id, repo_commit, repo_license,
                        language, file_path, symbol_name, start_line, end_line,
                        snippet, match_rationale, match_confidence
                    )
                    VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7, $8,
                        $9, $10, $11, $12, $13,
                        $14, $15, $16
                    )
                    ON CONFLICT (tenant_id, workspace_id, algorithm_id, document_repo_id, file_path, start_line, end_line)
                    DO UPDATE SET
                        repo_commit = EXCLUDED.repo_commit,
                        repo_license = EXCLUDED.repo_license,
                        language = EXCLUDED.language,
                        symbol_name = EXCLUDED.symbol_name,
                        snippet = EXCLUDED.snippet,
                        match_rationale = EXCLUDED.match_rationale,
                        match_confidence = EXCLUDED.match_confidence,
                        updated_at = NOW()
                    RETURNING id, tenant_id, workspace_id, document_id,
                              algorithm_id, document_repo_id, repo_commit, repo_license,
                              language, file_path, symbol_name, start_line, end_line,
                              snippet, match_rationale, match_confidence,
                              status, embedding_id, created_at, updated_at
                    "#,
                )
                .bind(Uuid::new_v4())
                .bind(tenant_id)
                .bind(workspace_id)
                .bind(document_id)
                .bind(c.algorithm_id)
                .bind(c.document_repo_id)
                .bind(&c.repo_commit)
                .bind(c.repo_license.as_deref())
                .bind(&c.language)
                .bind(&c.file_path)
                .bind(c.symbol_name.as_deref())
                .bind(c.start_line)
                .bind(c.end_line)
                .bind(&c.snippet)
                .bind(c.match_rationale.as_deref())
                .bind(c.match_confidence.as_str())
                .fetch_one(&mut *tx)
                .await?;
                out.push(decode_artifact(row)?);
            }
            tx.commit().await?;
            Ok(out)
        }

        async fn list_for_document(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
        ) -> Result<Vec<CodeArtifact>, CodeStorageError> {
            let rows = sqlx::query(
                r#"
                SELECT id, tenant_id, workspace_id, document_id,
                       algorithm_id, document_repo_id, repo_commit, repo_license,
                       language, file_path, symbol_name, start_line, end_line,
                       snippet, match_rationale, match_confidence,
                       status, embedding_id, created_at, updated_at
                FROM code_artifacts
                WHERE tenant_id = $1 AND workspace_id = $2 AND document_id = $3
                ORDER BY algorithm_id,
                         CASE match_confidence WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                         created_at ASC
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .fetch_all(&self.pool)
            .await?;
            rows.into_iter().map(decode_artifact).collect()
        }

        async fn list_for_algorithm(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            algorithm_id: Uuid,
        ) -> Result<Vec<CodeArtifact>, CodeStorageError> {
            let rows = sqlx::query(
                r#"
                SELECT id, tenant_id, workspace_id, document_id,
                       algorithm_id, document_repo_id, repo_commit, repo_license,
                       language, file_path, symbol_name, start_line, end_line,
                       snippet, match_rationale, match_confidence,
                       status, embedding_id, created_at, updated_at
                FROM code_artifacts
                WHERE tenant_id = $1 AND workspace_id = $2 AND algorithm_id = $3
                ORDER BY CASE match_confidence WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                         created_at ASC
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(algorithm_id)
            .fetch_all(&self.pool)
            .await?;
            rows.into_iter().map(decode_artifact).collect()
        }

        async fn update_status(
            &self,
            id: Uuid,
            tenant_id: Uuid,
            workspace_id: Uuid,
            status: ArtifactStatus,
        ) -> Result<bool, CodeStorageError> {
            let res = sqlx::query(
                r#"
                UPDATE code_artifacts
                SET status = $1, updated_at = NOW()
                WHERE id = $2 AND tenant_id = $3 AND workspace_id = $4
                "#,
            )
            .bind(status.as_str())
            .bind(id)
            .bind(tenant_id)
            .bind(workspace_id)
            .execute(&self.pool)
            .await?;
            Ok(res.rows_affected() > 0)
        }

        async fn upsert_run(&self, run: &CodeReferenceRun) -> Result<(), CodeStorageError> {
            sqlx::query(
                r#"
                INSERT INTO code_reference_runs (
                    tenant_id, workspace_id, document_id, document_repo_id,
                    status, algorithm_count, finding_count, cost_usd_equivalent,
                    error_message, attempted_at, completed_at
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                ON CONFLICT (tenant_id, workspace_id, document_id, document_repo_id)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    algorithm_count = EXCLUDED.algorithm_count,
                    finding_count = EXCLUDED.finding_count,
                    cost_usd_equivalent = EXCLUDED.cost_usd_equivalent,
                    error_message = EXCLUDED.error_message,
                    attempted_at = EXCLUDED.attempted_at,
                    completed_at = EXCLUDED.completed_at
                "#,
            )
            .bind(run.tenant_id)
            .bind(run.workspace_id)
            .bind(&run.document_id)
            .bind(run.document_repo_id)
            .bind(run.status.as_str())
            .bind(run.algorithm_count)
            .bind(run.finding_count)
            .bind(run.cost_usd_equivalent)
            .bind(run.error_message.as_deref())
            .bind(run.attempted_at)
            .bind(run.completed_at)
            .execute(&self.pool)
            .await?;
            Ok(())
        }

        async fn mark_run_status(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            document_repo_id: Uuid,
            status: RunStatus,
        ) -> Result<(), CodeStorageError> {
            sqlx::query(
                r#"
                INSERT INTO code_reference_runs (
                    tenant_id, workspace_id, document_id, document_repo_id,
                    status, algorithm_count, finding_count, attempted_at
                )
                VALUES ($1,$2,$3,$4,$5,0,0,NOW())
                ON CONFLICT (tenant_id, workspace_id, document_id, document_repo_id)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    error_message = NULL,
                    completed_at = NULL
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .bind(document_repo_id)
            .bind(status.as_str())
            .execute(&self.pool)
            .await?;
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        async fn mark_run_complete(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            document_repo_id: Uuid,
            algorithm_count: i32,
            finding_count: i32,
            cost_usd_equivalent: Option<f32>,
        ) -> Result<(), CodeStorageError> {
            sqlx::query(
                r#"
                UPDATE code_reference_runs
                SET status = CASE WHEN $5 > 0 THEN 'awaiting_review' ELSE 'complete' END,
                    algorithm_count = $5,
                    finding_count = $6,
                    cost_usd_equivalent = $7,
                    error_message = NULL,
                    completed_at = NOW()
                WHERE tenant_id = $1 AND workspace_id = $2
                  AND document_id = $3 AND document_repo_id = $4
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .bind(document_repo_id)
            .bind(algorithm_count)
            .bind(finding_count)
            .bind(cost_usd_equivalent)
            .execute(&self.pool)
            .await?;
            Ok(())
        }

        async fn mark_run_failed(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            document_repo_id: Uuid,
            error_message: &str,
        ) -> Result<(), CodeStorageError> {
            sqlx::query(
                r#"
                UPDATE code_reference_runs
                SET status = 'failed',
                    error_message = $5,
                    completed_at = NOW()
                WHERE tenant_id = $1 AND workspace_id = $2
                  AND document_id = $3 AND document_repo_id = $4
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .bind(document_repo_id)
            .bind(error_message)
            .execute(&self.pool)
            .await?;
            Ok(())
        }

        async fn get_run(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            document_repo_id: Uuid,
        ) -> Result<Option<CodeReferenceRun>, CodeStorageError> {
            let row = sqlx::query(
                r#"
                SELECT tenant_id, workspace_id, document_id, document_repo_id,
                       status, algorithm_count, finding_count, cost_usd_equivalent,
                       error_message, attempted_at, completed_at
                FROM code_reference_runs
                WHERE tenant_id = $1 AND workspace_id = $2
                  AND document_id = $3 AND document_repo_id = $4
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .bind(document_repo_id)
            .fetch_optional(&self.pool)
            .await?;
            row.map(decode_run).transpose()
        }

        async fn counts_by_document(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
        ) -> Result<Vec<(String, i64)>, CodeStorageError> {
            // Count a document as "having a reference" if it has EITHER
            // `code_artifacts` rows (extracted implementations) OR
            // `document_repos` rows (detected/approved reference repos
            // awaiting analysis). Without the document_repos side, a doc
            // whose repo has been detected but whose code-analyzer run
            // hasn't landed yet would hide the row-level button and the
            // user has no entry point from the doc list into the Code
            // Matches tab — where they can approve the repo and trigger
            // analysis.
            let rows: Vec<(String, i64)> = sqlx::query_as(
                r#"
                SELECT document_id, SUM(c)::BIGINT AS total FROM (
                    SELECT document_id, COUNT(*)::BIGINT AS c
                    FROM code_artifacts
                    WHERE tenant_id = $1 AND workspace_id = $2
                    GROUP BY document_id
                    UNION ALL
                    SELECT document_id, COUNT(*)::BIGINT AS c
                    FROM document_repos
                    WHERE tenant_id = $1 AND workspace_id = $2
                    GROUP BY document_id
                ) merged
                GROUP BY document_id
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .fetch_all(&self.pool)
            .await?;
            Ok(rows)
        }
    }
}
