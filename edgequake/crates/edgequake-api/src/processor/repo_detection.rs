//! Reference-repository detection task processor.
//!
//! Runs Phase 0 of the Reference Code GraphRAG extension:
//!   Layer A — `edgequake-pdf`: hyperlinks + reference-section filter.
//!   Layer B — `edgequake-agents`: SearXNG + Crawl4AI + LLM fallback when A finds nothing.
//!
//! Exposed two ways:
//! - As a standalone `TaskType::RepoDetection` task (enqueued by API, or later by
//!   a "re-detect" button in the UI).
//! - As an inline helper `run_repo_detection_inline` callable from other
//!   processors (e.g. PDF processing runs it at the end of conversion).

use super::*;
use edgequake_agents::repo_detection::{
    run_detection, DetectionOutcome, PostgresRepoStorage, RepoDetectionConfig, RepoStorage,
    VerificationVerdict, WebSearchClients,
};
use edgequake_agents::web_search::{Crawl4aiClient, GithubClient, SearxngClient};
use tokio_util::sync::CancellationToken;

impl DocumentTaskProcessor {
    /// Public TaskProcessor dispatch entry — loads PDF + markdown from storage
    /// and runs detection.
    #[cfg(feature = "postgres")]
    pub(super) async fn process_repo_detection(
        &self,
        task: &mut Task,
        data: edgequake_tasks::RepoDetectionData,
        cancel_token: CancellationToken,
    ) -> TaskResult<serde_json::Value> {
        let document_id = &data.document_id;

        info!(
            document_id = %document_id,
            workspace_id = %data.workspace_id,
            pdf_id = ?data.pdf_id,
            "Processing repo-detection task"
        );

        self.check_cancelled(&cancel_token, "repo_detecting", document_id)
            .await?;
        task.update_progress("repo_detecting".to_string(), 1, 10);

        // Load PDF bytes AND the persisted markdown from `pdf_documents` in one
        // fetch when a pdf_id is provided. Markdown lives on the row
        // (`markdown_content`) since pipeline refactor; the old `{doc_id}-content`
        // KV key is no longer populated, so Layer B was silently running with an
        // empty front-matter input for every PDF and short-circuiting with
        // `NoFrontMatter`.
        let (pdf_bytes, markdown): (Vec<u8>, String) = match data.pdf_id.as_ref() {
            Some(pdf_id) => {
                let uuid = uuid::Uuid::parse_str(pdf_id)
                    .map_err(|e| TaskError::Process(format!("Invalid pdf_id: {e}")))?;
                let store = self
                    .pdf_storage
                    .as_ref()
                    .ok_or_else(|| TaskError::Process("PDF storage not available".into()))?;
                let pdf = store
                    .get_pdf(&uuid)
                    .await
                    .map_err(|e| TaskError::Process(format!("Failed to fetch PDF: {e}")))?
                    .ok_or_else(|| TaskError::NotFound(format!("PDF not found: {pdf_id}")))?;
                (pdf.pdf_data, pdf.markdown_content.unwrap_or_default())
            }
            None => (Vec::new(), String::new()),
        };

        let outcome = self
            .run_repo_detection_inline(
                task.tenant_id,
                task.workspace_id,
                document_id,
                &pdf_bytes,
                &markdown,
            )
            .await?;

        task.update_progress("completed".to_string(), 1, 100);
        Ok(json!({
            "document_id": document_id,
            "layer_a_candidates": outcome.layer_a_count(),
            "layer_b_candidates": outcome.layer_b_count(),
            "total_candidates": outcome.candidates.len(),
            "layer_b_attempted": outcome.layer_b_attempted,
        }))
    }

    /// Shared helper: run detection, persist candidates + detection-run state.
    ///
    /// Safe to call from other processors (e.g. `process_pdf_processing`) as a
    /// fire-and-nearly-forget step — failures are swallowed after being
    /// recorded to `document_repo_detections`, so they don't fail the parent
    /// task.
    #[cfg(feature = "postgres")]
    pub(super) async fn run_repo_detection_inline(
        &self,
        tenant_id: uuid::Uuid,
        workspace_id: uuid::Uuid,
        document_id: &str,
        pdf_bytes: &[u8],
        markdown: &str,
    ) -> TaskResult<DetectionOutcome> {
        let storage = match self.repo_storage().await {
            Ok(s) => s,
            Err(e) => return Err(TaskError::Process(format!("repo storage: {e}"))),
        };
        storage
            .mark_detection_running(tenant_id, workspace_id, document_id)
            .await
            .ok();

        let web_clients = self.build_web_search_clients(workspace_id).await;
        let config = RepoDetectionConfig::default();

        let outcome_result =
            run_detection(pdf_bytes, markdown, web_clients.as_ref(), &config).await;

        let mut outcome = match outcome_result {
            Ok(o) => o,
            Err(e) => {
                warn!(document_id, error = %e, "repo detection failed");
                storage
                    .mark_detection_failed(tenant_id, workspace_id, document_id, &e.to_string())
                    .await
                    .ok();
                return Err(TaskError::Process(format!("repo detection failed: {e}")));
            }
        };

        // Workspace-level filter: when `accept_unofficial_implementations`
        // is off (default), drop candidates the verifier classified as
        // `third_party` or `unrelated` before persistence so they never
        // show up in the review queue. Candidates the verifier couldn't
        // score (verification = None) OR classified as `official` /
        // `inconclusive` still pass through — we don't auto-reject on
        // uncertainty, consistent with the existing verifier-failure
        // policy.
        let accept_unofficial = self.resolve_accept_unofficial(workspace_id).await;
        if !accept_unofficial {
            let before = outcome.candidates.len();
            outcome.candidates.retain(|c| match &c.verification {
                Some(v) => !matches!(
                    v.verdict,
                    VerificationVerdict::ThirdParty | VerificationVerdict::Unrelated
                ),
                None => true,
            });
            let dropped = before - outcome.candidates.len();
            if dropped > 0 {
                info!(
                    document_id,
                    dropped,
                    "accept_unofficial_implementations=false: dropped unofficial candidates"
                );
            }
        }

        info!(
            document_id,
            layer_a = outcome.layer_a_count(),
            layer_b = outcome.layer_b_count(),
            accept_unofficial,
            "repo detection outcome"
        );

        if let Err(e) = storage
            .upsert_candidates(tenant_id, workspace_id, document_id, &outcome.candidates)
            .await
        {
            warn!(document_id, error = %e, "failed to persist repo candidates");
            storage
                .mark_detection_failed(tenant_id, workspace_id, document_id, &e.to_string())
                .await
                .ok();
            return Err(TaskError::Storage(format!("persist candidates: {e}")));
        }

        storage
            .mark_detection_complete(
                tenant_id,
                workspace_id,
                document_id,
                outcome.layer_a_count() as i32,
                outcome.layer_b_count() as i32,
            )
            .await
            .ok();

        Ok(outcome)
    }

    /// Look up the workspace setting that gates non-official candidates.
    /// Defaults to `false` when the workspace-service isn't wired or the
    /// workspace row is unreachable — safer to drop unofficial than to
    /// silently persist them.
    async fn resolve_accept_unofficial(&self, workspace_id: uuid::Uuid) -> bool {
        let Some(ws_svc) = self.workspace_service.as_ref() else {
            return false;
        };
        match ws_svc.get_workspace(workspace_id).await {
            Ok(Some(ws)) => ws.accept_unofficial_implementations.unwrap_or(false),
            Ok(None) => {
                warn!(
                    workspace_id = %workspace_id,
                    "workspace not found while resolving accept_unofficial_implementations; \
                     defaulting to false"
                );
                false
            }
            Err(e) => {
                warn!(
                    workspace_id = %workspace_id,
                    error = %e,
                    "failed to fetch workspace setting; defaulting to accept_unofficial=false"
                );
                false
            }
        }
    }

    /// Build a fresh PostgresRepoStorage using the task's DATABASE_URL. The
    /// cost of opening a second pool is trivial; sharing the API's pool would
    /// require plumbing it through `DocumentTaskProcessor` which isn't worth
    /// the churn for one feature.
    #[cfg(feature = "postgres")]
    async fn repo_storage(&self) -> Result<PostgresRepoStorage, String> {
        let url = std::env::var("DATABASE_URL")
            .map_err(|_| "DATABASE_URL not set for repo detection".to_string())?;
        let pool = sqlx::PgPool::connect(&url)
            .await
            .map_err(|e| format!("connect postgres: {e}"))?;
        Ok(PostgresRepoStorage::new(pool))
    }

    /// Build Layer B clients. The search arm is **GitHub** now — the
    /// SearXNG/Crawl4AI path stayed compiled because the verifier still
    /// uses Crawl4AI for non-GitHub README fetches, but they no longer
    /// drive repo discovery. Returns `None` only when no LLM is reachable;
    /// SearXNG/Crawl4AI are optional and we substitute no-op URLs if
    /// their env vars are unset.
    ///
    /// The verifier + resolver LLM is resolved from the workspace's
    /// configured `llm_provider`/`llm_model` so repo-detection talks to
    /// the same model the rest of the pipeline uses for that workspace.
    /// Falls back to `self.llm_provider` (server default from env at
    /// startup) when workspace lookup fails — the alternative is
    /// hard-failing detection on a transient DB blip.
    async fn build_web_search_clients(
        &self,
        workspace_id: uuid::Uuid,
    ) -> Option<WebSearchClients> {
        let llm = self.resolve_workspace_llm(workspace_id).await;
        let github = GithubClient::from_env();
        if !github.authenticated {
            warn!(
                "GITHUB_TOKEN not set — Layer B GitHub Search runs unauthenticated \
                 (60 req/hr). Set GITHUB_TOKEN to lift the cap to 5000/hr."
            );
        }
        // Crawl4AI/SearXNG are optional — substitute placeholder URLs so
        // the verifier's "fall back to crawl4ai for non-github readme"
        // path returns a clean error instead of panicking.
        let crawl4ai_url = std::env::var("CRAWL4AI_URL")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "http://localhost:0".to_string());
        let searxng_url = std::env::var("SEARXNG_URL")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "http://localhost:0".to_string());
        Some(WebSearchClients {
            github,
            searxng: SearxngClient::new(searxng_url),
            crawl4ai: Crawl4aiClient::new(crawl4ai_url),
            llm,
        })
    }

    /// Build a workspace-bound LLM client for repo-detection. Mirrors the
    /// resolution chain used by `algorithm_extraction::resolve_algorithm_llm_providers`:
    /// workspace's `llm_provider`+`llm_model` → server default. Logs the
    /// outcome so misconfiguration is easy to spot in tracing.
    async fn resolve_workspace_llm(
        &self,
        workspace_id: uuid::Uuid,
    ) -> std::sync::Arc<dyn edgequake_llm::traits::LLMProvider> {
        use crate::safety_limits::create_safe_llm_provider;

        let default = std::sync::Arc::clone(&self.llm_provider);
        let Some(ws_svc) = self.workspace_service.as_ref() else {
            return default;
        };
        match ws_svc.get_workspace(workspace_id).await {
            Ok(Some(ws)) => match create_safe_llm_provider(&ws.llm_provider, &ws.llm_model) {
                Ok(p) => {
                    info!(
                        workspace_id = %workspace_id,
                        provider = %ws.llm_provider,
                        model = %ws.llm_model,
                        "repo-detection: using workspace LLM"
                    );
                    p
                }
                Err(e) => {
                    warn!(
                        workspace_id = %workspace_id,
                        provider = %ws.llm_provider,
                        model = %ws.llm_model,
                        error = %e,
                        "repo-detection: workspace LLM init failed; falling back to server default"
                    );
                    default
                }
            },
            Ok(None) => {
                warn!(
                    workspace_id = %workspace_id,
                    "repo-detection: workspace not found; falling back to server-default LLM"
                );
                default
            }
            Err(e) => {
                warn!(
                    workspace_id = %workspace_id,
                    error = %e,
                    "repo-detection: failed to fetch workspace; falling back to server-default LLM"
                );
                default
            }
        }
    }
}
