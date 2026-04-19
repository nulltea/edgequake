//! Reference-code analysis task processor (Phase 1).
//!
//! Orchestrates the per-approved-repo pipeline:
//!
//! 1. Load the approved `document_repos` row by id.
//! 2. Load all `approved` algorithms for this document.
//! 3. Mark the `code_reference_runs` row as `analyzing` and POST to
//!    the code-analyzer sidecar.
//! 4. For each finding, read the snippet from the shared `/workspace`
//!    volume, trim to a bounded size, upsert a `code_artifacts` row.
//! 5. Mark run `awaiting_review` (or `complete` if zero findings).
//!
//! Embedding of approved snippets is a separate step triggered on the
//! user's "approve" click — kept out of this processor so failures in the
//! embedding backend don't block the localization signal reaching the UI.

use super::*;
use edgequake_agents::code_analysis::{
    extract_snippet, language_from_path, AnalyzerAlgorithmInput, AnalyzerClient,
    AnalyzerClientError, AnalyzerRequest, CodeArtifactCandidate, CodeArtifactStorage,
    MatchConfidence, PostgresCodeArtifactStorage, RunStatus,
};
use edgequake_agents::repo_detection::{PostgresRepoStorage, RepoStorage};
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;

impl DocumentTaskProcessor {
    /// TaskProcessor dispatch entry.
    #[cfg(feature = "postgres")]
    pub(super) async fn process_code_reference_analysis(
        &self,
        task: &mut Task,
        data: edgequake_tasks::CodeReferenceAnalysisData,
        cancel_token: CancellationToken,
    ) -> TaskResult<serde_json::Value> {
        let document_id = &data.document_id;
        let repo_id = data.document_repo_id;

        info!(
            document_id = %document_id,
            document_repo_id = %repo_id,
            "Processing code-reference analysis task"
        );

        self.check_cancelled(&cancel_token, "code_ref_cloning", document_id)
            .await?;

        let tenant_id = task.tenant_id;
        let workspace_id = task.workspace_id;

        let code_storage = match self.code_storage().await {
            Ok(s) => s,
            Err(e) => return Err(TaskError::Process(format!("code storage: {e}"))),
        };
        let repo_storage = match self.repo_detection_storage().await {
            Ok(s) => s,
            Err(e) => return Err(TaskError::Process(format!("repo storage: {e}"))),
        };

        // ── Step 1: Resolve the approved repo row. ────────────────────────
        // Only one doc_repos row per (doc, url) — list_for_document is the
        // simplest way until we add a get_by_id helper.
        let repo = repo_storage
            .list_for_document(tenant_id, workspace_id, document_id)
            .await
            .map_err(|e| TaskError::Storage(format!("load document_repos: {e}")))?
            .into_iter()
            .find(|r| r.id == repo_id)
            .ok_or_else(|| {
                TaskError::NotFound(format!(
                    "document_repos row {repo_id} not found for doc {document_id}"
                ))
            })?;

        // ── Step 2: Load approved algorithms for this doc. ────────────────
        let algorithms = self
            .load_approved_algorithms(tenant_id, workspace_id, document_id)
            .await?;

        if algorithms.is_empty() {
            info!(
                document_id = %document_id,
                "No approved algorithms — skipping code-reference analysis"
            );
            code_storage
                .mark_run_complete(tenant_id, workspace_id, document_id, repo_id, 0, 0, None)
                .await
                .ok();
            return Ok(json!({
                "document_id": document_id,
                "document_repo_id": repo_id,
                "status": "skipped_no_algorithms",
            }));
        }

        // ── Step 3: analyzer call (cloning + analyzing phases) ────────────
        code_storage
            .mark_run_status(
                tenant_id,
                workspace_id,
                document_id,
                repo_id,
                RunStatus::Cloning,
            )
            .await
            .ok();
        task.update_progress("code_ref_cloning".to_string(), 4, 10);

        let analyzer_url = std::env::var("CODE_ANALYZER_URL")
            .unwrap_or_else(|_| "http://code-analyzer:9100".to_string());
        let analyzer = AnalyzerClient::new(analyzer_url);

        code_storage
            .mark_run_status(
                tenant_id,
                workspace_id,
                document_id,
                repo_id,
                RunStatus::Analyzing,
            )
            .await
            .ok();
        task.update_progress("code_ref_analyzing".to_string(), 4, 30);

        let analyzer_req = AnalyzerRequest {
            repo_url: repo.url.clone(),
            repo_commit: "HEAD".to_string(),
            algorithms: algorithms
                .iter()
                .map(|a| AnalyzerAlgorithmInput {
                    id: a.id.to_string(),
                    name: a.name.clone(),
                    description: a.description.clone().unwrap_or_default(),
                    pseudocode: a.pseudocode.clone(),
                    languages_hint: Vec::new(),
                })
                .collect(),
            size_cap_mb: std::env::var("EDGEQUAKE_CODE_REFERENCE_MAX_REPO_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(500),
            timeout_s: std::env::var("EDGEQUAKE_CODE_REFERENCE_ANALYZER_TIMEOUT_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600),
            model: std::env::var("EDGEQUAKE_CODE_ANALYZER_MODEL").ok(),
        };

        let analyzer_resp = match analyzer.analyze(&analyzer_req).await {
            Ok(r) => r,
            Err(AnalyzerClientError::Status { status, body }) => {
                let msg = format!("code-analyzer returned {status}: {body}");
                warn!(document_id, error = %msg, "analyzer call failed");
                code_storage
                    .mark_run_failed(tenant_id, workspace_id, document_id, repo_id, &msg)
                    .await
                    .ok();
                return Err(TaskError::Process(msg));
            }
            Err(e) => {
                let msg = format!("code-analyzer error: {e}");
                warn!(document_id, error = %msg, "analyzer call failed");
                code_storage
                    .mark_run_failed(tenant_id, workspace_id, document_id, repo_id, &msg)
                    .await
                    .ok();
                return Err(TaskError::Process(msg));
            }
        };

        info!(
            document_id = %document_id,
            findings = analyzer_resp.findings.len(),
            repo_commit = %analyzer_resp.repo_commit,
            cost_usd_equivalent = ?analyzer_resp.usage_cost_usd_equivalent,
            "analyzer returned findings"
        );

        // ── Step 4: snippet extraction → candidates → upsert. ─────────────
        code_storage
            .mark_run_status(
                tenant_id,
                workspace_id,
                document_id,
                repo_id,
                RunStatus::Embedding,
            )
            .await
            .ok();
        task.update_progress("code_ref_extracting".to_string(), 4, 60);

        let repo_root = PathBuf::from(&analyzer_resp.repo_path);
        let mut candidates = Vec::with_capacity(analyzer_resp.findings.len());
        let mut dropped = 0usize;
        for f in &analyzer_resp.findings {
            let algo_uuid = match uuid::Uuid::parse_str(&f.algorithm_id) {
                Ok(u) => u,
                Err(_) => {
                    warn!(
                        finding_algo = %f.algorithm_id,
                        "analyzer returned non-UUID algorithm_id, dropping"
                    );
                    dropped += 1;
                    continue;
                }
            };
            let snippet = match extract_snippet(&repo_root, &f.file, f.start_line, f.end_line) {
                Ok(s) => s,
                Err(e) => {
                    warn!(
                        file = %f.file,
                        start = f.start_line,
                        end = f.end_line,
                        error = %e,
                        "snippet extraction failed, dropping finding"
                    );
                    dropped += 1;
                    continue;
                }
            };
            let confidence =
                MatchConfidence::parse(&f.confidence).unwrap_or(MatchConfidence::Medium);
            candidates.push(CodeArtifactCandidate {
                algorithm_id: algo_uuid,
                document_repo_id: repo_id,
                repo_commit: analyzer_resp.repo_commit.clone(),
                repo_license: analyzer_resp.repo_license.clone(),
                language: language_from_path(&f.file).to_string(),
                file_path: f.file.clone(),
                symbol_name: None,
                start_line: f.start_line,
                end_line: f.end_line,
                snippet: snippet.text,
                match_rationale: Some(f.rationale.clone()),
                match_confidence: confidence,
            });
        }

        if dropped > 0 {
            warn!(
                document_id = %document_id,
                dropped,
                total = analyzer_resp.findings.len(),
                "dropped analyzer findings due to validation errors"
            );
        }

        code_storage
            .upsert_candidates(tenant_id, workspace_id, document_id, &candidates)
            .await
            .map_err(|e| TaskError::Storage(format!("persist code_artifacts: {e}")))?;

        // ── Step 5: close the run. ────────────────────────────────────────
        code_storage
            .mark_run_complete(
                tenant_id,
                workspace_id,
                document_id,
                repo_id,
                algorithms.len() as i32,
                candidates.len() as i32,
                analyzer_resp.usage_cost_usd_equivalent,
            )
            .await
            .map_err(|e| TaskError::Storage(format!("close run: {e}")))?;

        task.update_progress("completed".to_string(), 4, 100);
        Ok(json!({
            "document_id": document_id,
            "document_repo_id": repo_id,
            "repo_commit": analyzer_resp.repo_commit,
            "algorithm_count": algorithms.len(),
            "finding_count": candidates.len(),
            "dropped": dropped,
            "cost_usd_equivalent": analyzer_resp.usage_cost_usd_equivalent,
            "duration_ms": analyzer_resp.duration_ms,
            "num_turns": analyzer_resp.num_turns,
        }))
    }

    #[cfg(feature = "postgres")]
    async fn code_storage(&self) -> Result<PostgresCodeArtifactStorage, String> {
        let url = std::env::var("DATABASE_URL").map_err(|_| "DATABASE_URL not set".to_string())?;
        let pool = sqlx::PgPool::connect(&url)
            .await
            .map_err(|e| format!("connect postgres: {e}"))?;
        Ok(PostgresCodeArtifactStorage::new(pool))
    }

    #[cfg(feature = "postgres")]
    async fn repo_detection_storage(&self) -> Result<PostgresRepoStorage, String> {
        let url = std::env::var("DATABASE_URL").map_err(|_| "DATABASE_URL not set".to_string())?;
        let pool = sqlx::PgPool::connect(&url)
            .await
            .map_err(|e| format!("connect postgres: {e}"))?;
        Ok(PostgresRepoStorage::new(pool))
    }

    #[cfg(feature = "postgres")]
    async fn load_approved_algorithms(
        &self,
        tenant_id: uuid::Uuid,
        workspace_id: uuid::Uuid,
        document_id: &str,
    ) -> TaskResult<Vec<edgequake_algorithms::Algorithm>> {
        use edgequake_algorithms::{AlgorithmStatus, AlgorithmStorage, PostgresAlgorithmStorage};

        let url = std::env::var("DATABASE_URL")
            .map_err(|_| TaskError::Process("DATABASE_URL not set".to_string()))?;
        let pool = sqlx::PgPool::connect(&url)
            .await
            .map_err(|e| TaskError::Storage(format!("connect postgres: {e}")))?;
        let algos = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool));

        algos
            .list_algorithms(
                document_id,
                tenant_id,
                workspace_id,
                Some(AlgorithmStatus::Approved),
            )
            .await
            .map_err(|e| TaskError::Storage(format!("list algorithms: {e}")))
    }
}
