//! Storage for `document_repos` + `document_repo_detections`.
//!
//! Runtime-checked sqlx queries (same pattern as `edgequake-algorithms`).
//! Postgres-only impl lives under the `postgres` feature.

use async_trait::async_trait;
use uuid::Uuid;

use super::types::{DetectionRun, DetectionRunStatus, DocumentRepo, RepoCandidate, RepoStatus};

#[derive(Debug, thiserror::Error)]
pub enum RepoStorageError {
    #[cfg(feature = "postgres")]
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("decode error: {0}")]
    Decode(String),
}

#[async_trait]
pub trait RepoStorage: Send + Sync {
    /// Persist candidates idempotently: `ON CONFLICT` on
    /// `(tenant, workspace, document, host, owner, repo)` updates the
    /// timestamp and leaves the human-review status alone.
    async fn upsert_candidates(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        candidates: &[RepoCandidate],
    ) -> Result<(), RepoStorageError>;

    /// List all candidates attached to a document.
    async fn list_for_document(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
    ) -> Result<Vec<DocumentRepo>, RepoStorageError>;

    /// Update a single candidate's review status.
    async fn update_status(
        &self,
        id: Uuid,
        tenant_id: Uuid,
        workspace_id: Uuid,
        status: RepoStatus,
    ) -> Result<bool, RepoStorageError>;

    /// Record that a detection attempt has started (idempotent: upserts the
    /// single row keyed by `(tenant, workspace, document)`).
    async fn mark_detection_running(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
    ) -> Result<(), RepoStorageError>;

    /// Record a successful detection run (sets counts + completed_at).
    async fn mark_detection_complete(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        layer_a_count: i32,
        layer_b_count: i32,
    ) -> Result<(), RepoStorageError>;

    /// Record a failed detection run (keeps counts, stores error).
    async fn mark_detection_failed(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        error_message: &str,
    ) -> Result<(), RepoStorageError>;

    /// Fetch the detection-run state for a document, if any.
    async fn get_detection_run(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
    ) -> Result<Option<DetectionRun>, RepoStorageError>;

    /// Delete a single candidate. Used by the review flow when a user
    /// rejects a repo — rather than keep a dangling rejected row, we
    /// drop it (and cascade to `code_artifacts` via the FK).
    async fn delete(
        &self,
        id: Uuid,
        tenant_id: Uuid,
        workspace_id: Uuid,
    ) -> Result<bool, RepoStorageError>;

    /// Insert a single candidate directly, bypassing detection. Used by the
    /// "Add reference manually" UI path on the References tab.
    async fn insert_manual(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        url: &str,
    ) -> Result<DocumentRepo, RepoStorageError>;
}

#[cfg(feature = "postgres")]
pub use postgres::PostgresRepoStorage;

#[cfg(feature = "postgres")]
mod postgres {
    use super::*;
    use crate::repo_detection::types::{Confidence, DetectionMethod, RepoHost};
    use crate::repo_detection::verify::{VerificationReport, VerificationVerdict};
    use chrono::{DateTime, Utc};
    use sqlx::{postgres::PgRow, PgPool, Row};

    pub struct PostgresRepoStorage {
        pool: PgPool,
    }

    impl PostgresRepoStorage {
        pub fn new(pool: PgPool) -> Self {
            Self { pool }
        }
    }

    fn decode_row(row: PgRow) -> Result<DocumentRepo, RepoStorageError> {
        let host_s: String = row.try_get("host")?;
        let method_s: String = row.try_get("detection_method")?;
        let confidence_s: String = row.try_get("confidence")?;
        let status_s: String = row.try_get("status")?;

        let host = RepoHost::parse(&host_s)
            .ok_or_else(|| RepoStorageError::Decode(format!("bad host: {host_s}")))?;
        let detection_method = DetectionMethod::parse(&method_s)
            .ok_or_else(|| RepoStorageError::Decode(format!("bad method: {method_s}")))?;
        let confidence = Confidence::parse(&confidence_s)
            .ok_or_else(|| RepoStorageError::Decode(format!("bad confidence: {confidence_s}")))?;
        let status = RepoStatus::parse(&status_s)
            .ok_or_else(|| RepoStorageError::Decode(format!("bad status: {status_s}")))?;

        // Migration 050 — verification columns. All nullable; decode lazily.
        let verification_verdict: Option<String> = row.try_get("verification_verdict")?;
        let verification_confidence: Option<f32> = row.try_get("verification_confidence")?;
        let verification_rationale: Option<String> = row.try_get("verification_rationale")?;
        let verified_at: Option<DateTime<Utc>> = row.try_get("verified_at")?;

        let verification = verification_verdict.and_then(|v| {
            Some(VerificationReport {
                verdict: VerificationVerdict::parse(&v)?,
                confidence: verification_confidence.unwrap_or(0.0).clamp(0.0, 1.0),
                rationale: verification_rationale.unwrap_or_default(),
            })
        });

        Ok(DocumentRepo {
            id: row.try_get("id")?,
            tenant_id: row.try_get("tenant_id")?,
            workspace_id: row.try_get("workspace_id")?,
            document_id: row.try_get("document_id")?,
            host,
            owner: row.try_get("owner")?,
            repo: row.try_get("repo")?,
            url: row.try_get("url")?,
            detection_method,
            pdf_page_index: row.try_get("pdf_page_index")?,
            search_rank: row.try_get("search_rank")?,
            source_url: row.try_get("source_url")?,
            confidence,
            status,
            created_at: row.try_get::<DateTime<Utc>, _>("created_at")?,
            updated_at: row.try_get::<DateTime<Utc>, _>("updated_at")?,
            verification,
            verified_at,
        })
    }

    fn decode_run(row: PgRow) -> Result<DetectionRun, RepoStorageError> {
        let status_s: String = row.try_get("status")?;
        let status = match status_s.as_str() {
            "running" => DetectionRunStatus::Running,
            "complete" => DetectionRunStatus::Complete,
            "failed" => DetectionRunStatus::Failed,
            other => return Err(RepoStorageError::Decode(format!("bad run status: {other}"))),
        };
        Ok(DetectionRun {
            tenant_id: row.try_get("tenant_id")?,
            workspace_id: row.try_get("workspace_id")?,
            document_id: row.try_get("document_id")?,
            status,
            layer_a_candidates: row.try_get("layer_a_candidates")?,
            layer_b_candidates: row.try_get("layer_b_candidates")?,
            error_message: row.try_get("error_message")?,
            attempted_at: row.try_get::<DateTime<Utc>, _>("attempted_at")?,
            completed_at: row.try_get::<Option<DateTime<Utc>>, _>("completed_at")?,
        })
    }

    #[async_trait]
    impl RepoStorage for PostgresRepoStorage {
        async fn upsert_candidates(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            candidates: &[RepoCandidate],
        ) -> Result<(), RepoStorageError> {
            if candidates.is_empty() {
                return Ok(());
            }
            let mut tx = self.pool.begin().await?;
            for c in candidates {
                let v_verdict = c.verification.as_ref().map(|v| v.verdict.as_str());
                let v_confidence = c.verification.as_ref().map(|v| v.confidence);
                let v_rationale = c.verification.as_ref().map(|v| v.rationale.as_str());
                // Only stamp verified_at when we actually have a verification
                // report — NULL otherwise so the column tracks "did we run
                // the verifier for this row".
                let v_time = c.verification.as_ref().map(|_| chrono::Utc::now());

                sqlx::query(
                    r#"
                    INSERT INTO document_repos (
                        id, tenant_id, workspace_id, document_id,
                        host, owner, repo, url,
                        detection_method, pdf_page_index, search_rank, source_url,
                        confidence, status,
                        verification_verdict, verification_confidence,
                        verification_rationale, verified_at
                    )
                    VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7, $8,
                        $9, $10, $11, $12,
                        $13, 'pending',
                        $14, $15, $16, $17
                    )
                    ON CONFLICT (tenant_id, workspace_id, document_id, host, owner, repo)
                    DO UPDATE SET
                        url = EXCLUDED.url,
                        detection_method = EXCLUDED.detection_method,
                        pdf_page_index = EXCLUDED.pdf_page_index,
                        search_rank = EXCLUDED.search_rank,
                        source_url = EXCLUDED.source_url,
                        confidence = EXCLUDED.confidence,
                        -- Overwrite verification fields on re-detect so the
                        -- row reflects the latest verifier run rather than
                        -- sticking with a stale verdict from a prior attempt.
                        verification_verdict = EXCLUDED.verification_verdict,
                        verification_confidence = EXCLUDED.verification_confidence,
                        verification_rationale = EXCLUDED.verification_rationale,
                        verified_at = EXCLUDED.verified_at,
                        updated_at = NOW()
                    "#,
                )
                .bind(Uuid::new_v4())
                .bind(tenant_id)
                .bind(workspace_id)
                .bind(document_id)
                .bind(c.host.as_str())
                .bind(&c.owner)
                .bind(&c.repo)
                .bind(&c.url)
                .bind(c.detection_method.as_str())
                .bind(c.pdf_page_index)
                .bind(c.search_rank)
                .bind(c.source_url.as_deref())
                .bind(c.confidence.as_str())
                .bind(v_verdict)
                .bind(v_confidence)
                .bind(v_rationale)
                .bind(v_time)
                .execute(&mut *tx)
                .await?;
            }
            tx.commit().await?;
            Ok(())
        }

        async fn list_for_document(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
        ) -> Result<Vec<DocumentRepo>, RepoStorageError> {
            let rows = sqlx::query(
                r#"
                SELECT id, tenant_id, workspace_id, document_id,
                       host, owner, repo, url,
                       detection_method, pdf_page_index, search_rank, source_url,
                       confidence, status, created_at, updated_at,
                       verification_verdict, verification_confidence,
                       verification_rationale, verified_at
                FROM document_repos
                WHERE tenant_id = $1 AND workspace_id = $2 AND document_id = $3
                ORDER BY
                    CASE confidence WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                    created_at ASC
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .fetch_all(&self.pool)
            .await?;

            rows.into_iter().map(decode_row).collect()
        }

        async fn update_status(
            &self,
            id: Uuid,
            tenant_id: Uuid,
            workspace_id: Uuid,
            status: RepoStatus,
        ) -> Result<bool, RepoStorageError> {
            let res = sqlx::query(
                r#"
                UPDATE document_repos
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

        async fn mark_detection_running(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
        ) -> Result<(), RepoStorageError> {
            sqlx::query(
                r#"
                INSERT INTO document_repo_detections (
                    tenant_id, workspace_id, document_id, status,
                    layer_a_candidates, layer_b_candidates, attempted_at
                )
                VALUES ($1, $2, $3, 'running', 0, 0, NOW())
                ON CONFLICT (tenant_id, workspace_id, document_id) DO UPDATE SET
                    status = 'running',
                    error_message = NULL,
                    attempted_at = NOW(),
                    completed_at = NULL
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .execute(&self.pool)
            .await?;
            Ok(())
        }

        async fn mark_detection_complete(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            layer_a_count: i32,
            layer_b_count: i32,
        ) -> Result<(), RepoStorageError> {
            sqlx::query(
                r#"
                UPDATE document_repo_detections
                SET status = 'complete',
                    layer_a_candidates = $4,
                    layer_b_candidates = $5,
                    error_message = NULL,
                    completed_at = NOW()
                WHERE tenant_id = $1 AND workspace_id = $2 AND document_id = $3
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .bind(layer_a_count)
            .bind(layer_b_count)
            .execute(&self.pool)
            .await?;
            Ok(())
        }

        async fn mark_detection_failed(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            error_message: &str,
        ) -> Result<(), RepoStorageError> {
            sqlx::query(
                r#"
                UPDATE document_repo_detections
                SET status = 'failed',
                    error_message = $4,
                    completed_at = NOW()
                WHERE tenant_id = $1 AND workspace_id = $2 AND document_id = $3
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .bind(error_message)
            .execute(&self.pool)
            .await?;
            Ok(())
        }

        async fn get_detection_run(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
        ) -> Result<Option<DetectionRun>, RepoStorageError> {
            let row = sqlx::query(
                r#"
                SELECT tenant_id, workspace_id, document_id, status,
                       layer_a_candidates, layer_b_candidates, error_message,
                       attempted_at, completed_at
                FROM document_repo_detections
                WHERE tenant_id = $1 AND workspace_id = $2 AND document_id = $3
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .fetch_optional(&self.pool)
            .await?;
            row.map(decode_run).transpose()
        }

        async fn delete(
            &self,
            id: Uuid,
            tenant_id: Uuid,
            workspace_id: Uuid,
        ) -> Result<bool, RepoStorageError> {
            let res = sqlx::query(
                r#"
                DELETE FROM document_repos
                WHERE id = $1 AND tenant_id = $2 AND workspace_id = $3
                "#,
            )
            .bind(id)
            .bind(tenant_id)
            .bind(workspace_id)
            .execute(&self.pool)
            .await?;
            Ok(res.rows_affected() > 0)
        }

        async fn insert_manual(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            url: &str,
        ) -> Result<DocumentRepo, RepoStorageError> {
            let (host, owner, repo) = parse_repo_url(url)
                .ok_or_else(|| RepoStorageError::Decode(format!("cannot parse repo URL: {url}")))?;
            let id = Uuid::new_v4();
            let row = sqlx::query(
                r#"
                INSERT INTO document_repos (
                    id, tenant_id, workspace_id, document_id,
                    host, owner, repo, url,
                    detection_method, pdf_page_index, search_rank, source_url,
                    confidence, status
                )
                VALUES (
                    $1, $2, $3, $4,
                    $5, $6, $7, $8,
                    'manual', NULL, NULL, NULL,
                    'high', 'pending'
                )
                ON CONFLICT (tenant_id, workspace_id, document_id, host, owner, repo)
                DO UPDATE SET
                    url = EXCLUDED.url,
                    detection_method = 'manual',
                    updated_at = NOW()
                RETURNING
                    id, tenant_id, workspace_id, document_id,
                    host, owner, repo, url,
                    detection_method, pdf_page_index, search_rank, source_url,
                    confidence, status, created_at, updated_at,
                    verification_verdict, verification_confidence,
                    verification_rationale, verified_at
                "#,
            )
            .bind(id)
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .bind(host.as_str())
            .bind(&owner)
            .bind(&repo)
            .bind(url)
            .fetch_one(&self.pool)
            .await?;
            decode_row(row)
        }
    }

    /// Parse a GitHub/GitLab/Bitbucket-style URL into `(host, owner, repo)`.
    /// Strips `https://`, `http://`, trailing `.git`, and anything past the
    /// `owner/repo` path. Returns `None` for anything we can't recognise.
    fn parse_repo_url(raw: &str) -> Option<(RepoHost, String, String)> {
        let s = raw.trim();
        let s = s
            .strip_prefix("https://")
            .or_else(|| s.strip_prefix("http://"))
            .unwrap_or(s);
        let s = s.strip_prefix("www.").unwrap_or(s);
        let (host_part, rest) = s.split_once('/')?;
        let host = if host_part.contains("github.com") {
            RepoHost::Github
        } else if host_part.contains("gitlab.com") {
            RepoHost::Gitlab
        } else if host_part.contains("bitbucket.org") {
            RepoHost::Bitbucket
        } else {
            return None;
        };
        let mut parts = rest.split('/').filter(|p| !p.is_empty());
        let owner = parts.next()?.to_string();
        let repo_raw = parts.next()?;
        let repo = repo_raw
            .strip_suffix(".git")
            .unwrap_or(repo_raw)
            .to_string();
        Some((host, owner, repo))
    }
}
