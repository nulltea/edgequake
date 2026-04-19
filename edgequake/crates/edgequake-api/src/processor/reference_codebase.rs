//! Full reference-codebase indexing task processor (Phase 2).

use super::*;
use edgequake_agents::{
    code_analysis::{
        AnalyzerClient, CodeArtifactStorage, JinaEmbedder, PostgresCodeArtifactStorage,
        SnapshotRequest,
    },
    reference_codebase::{
        CodebaseIndexMode, CodebaseIndexStatus, IndexLimits, PostgresReferenceCodebaseStorage,
        ReferenceCodebaseIndexer, ReferenceCodebaseStorage,
    },
    repo_detection::{PostgresRepoStorage, RepoStorage},
};
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;

impl DocumentTaskProcessor {
    #[cfg(feature = "postgres")]
    pub(super) async fn process_reference_codebase_index(
        &self,
        task: &mut Task,
        data: edgequake_tasks::ReferenceCodebaseIndexData,
        cancel_token: CancellationToken,
    ) -> TaskResult<serde_json::Value> {
        let tenant_id = task.tenant_id;
        let workspace_id = task.workspace_id;
        let document_id = data.document_id;
        let repo_id = data.document_repo_id;
        let mode = CodebaseIndexMode::parse(&data.mode).ok_or_else(|| {
            TaskError::InvalidPayload(format!(
                "reference codebase mode must be 'algorithm_focused' or 'full', got {}",
                data.mode
            ))
        })?;

        self.check_cancelled(&cancel_token, "reference_codebase_start", &document_id)
            .await?;
        task.update_progress("reference_codebase_start".to_string(), 5, 5);

        let pool = self.reference_codebase_pool().await?;
        let repo_storage = PostgresRepoStorage::new(pool.clone());
        let code_storage = PostgresCodeArtifactStorage::new(pool.clone());
        let ref_storage = PostgresReferenceCodebaseStorage::new(pool);

        let repo = repo_storage
            .list_for_document(tenant_id, workspace_id, &document_id)
            .await
            .map_err(|e| TaskError::Storage(format!("load document_repos: {e}")))?
            .into_iter()
            .find(|r| r.id == repo_id)
            .ok_or_else(|| {
                TaskError::NotFound(format!("document_repos row {repo_id} not found"))
            })?;

        if repo.status.as_str() != "approved" {
            return Err(TaskError::Process(format!(
                "reference codebase indexing requires an approved repo; repo {repo_id} is {}",
                repo.status.as_str()
            )));
        }

        let approved_artifacts = code_storage
            .list_for_document(tenant_id, workspace_id, &document_id)
            .await
            .map_err(|e| TaskError::Storage(format!("load code_artifacts: {e}")))?
            .into_iter()
            .filter(|a| a.document_repo_id == repo_id && a.status.as_str() == "approved")
            .collect::<Vec<_>>();

        if mode == CodebaseIndexMode::AlgorithmFocused && approved_artifacts.is_empty() {
            return Err(TaskError::Process(
                "algorithm_focused reference codebase indexing requires at least one approved code artifact"
                    .to_string(),
            ));
        }

        self.check_cancelled(&cancel_token, "reference_codebase_snapshot", &document_id)
            .await?;
        task.update_progress("reference_codebase_snapshot".to_string(), 5, 15);

        let analyzer_url = std::env::var("CODE_ANALYZER_URL")
            .unwrap_or_else(|_| "http://code-analyzer:9100".to_string());
        let analyzer = AnalyzerClient::new(analyzer_url);
        let snapshot = analyzer
            .snapshot(&SnapshotRequest {
                repo_url: repo.url.clone(),
                repo_commit: "HEAD".to_string(),
                size_cap_mb: std::env::var("EDGEQUAKE_REFERENCE_CODEBASE_MAX_REPO_MB")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(500),
                timeout_s: std::env::var("EDGEQUAKE_REFERENCE_CODEBASE_SNAPSHOT_TIMEOUT_S")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(300),
            })
            .await
            .map_err(|e| TaskError::Process(format!("code-analyzer snapshot failed: {e}")))?;

        let repo_root = PathBuf::from(&snapshot.repo_path);
        if !repo_root.is_dir() {
            return Err(TaskError::Process(format!(
                "code-analyzer returned missing repo path: {}",
                snapshot.repo_path
            )));
        }

        let index = ref_storage
            .start_index(
                tenant_id,
                workspace_id,
                &document_id,
                repo_id,
                &repo.url,
                &snapshot.repo_commit,
                &snapshot.repo_path,
                snapshot.repo_license.as_deref(),
                mode,
                data.force_reindex,
            )
            .await
            .map_err(|e| TaskError::Storage(format!("start reference codebase index: {e}")))?;

        ref_storage
            .mark_status(index.id, CodebaseIndexStatus::Scanning, None)
            .await
            .ok();
        self.check_cancelled(&cancel_token, "reference_codebase_scanning", &document_id)
            .await?;
        task.update_progress("reference_codebase_scanning".to_string(), 5, 30);

        // Honor the per-row max_files_override when set. Populated by
        // the Phase 2 auto-enqueue path with a conservative cap (~5k);
        // explicit POST /indexes calls leave it NULL and fall back to
        // the global EDGEQUAKE_REFERENCE_CODEBASE_MAX_FILES env.
        let mut limits = IndexLimits::default();
        if let Some(cap) = read_max_files_override(index.id, &ref_storage).await {
            limits.max_files = limits.max_files.min(cap as usize);
            tracing::info!(
                index_id = %index.id,
                capped_max_files = cap,
                "reference-codebase indexer honouring max_files_override"
            );
        }

        let build = match ReferenceCodebaseIndexer::new(limits).build(
            &repo_root,
            mode,
            &approved_artifacts,
        ) {
            Ok(output) => output,
            Err(e) => {
                let msg = e.to_string();
                ref_storage
                    .mark_status(index.id, CodebaseIndexStatus::Failed, Some(&msg))
                    .await
                    .ok();
                return Err(TaskError::Process(msg));
            }
        };

        ref_storage
            .mark_status(index.id, CodebaseIndexStatus::Chunking, None)
            .await
            .ok();
        self.check_cancelled(&cancel_token, "reference_codebase_persisting", &document_id)
            .await?;
        task.update_progress("reference_codebase_persisting".to_string(), 5, 55);

        ref_storage
            .replace_index_data(&index, &build)
            .await
            .map_err(|e| TaskError::Storage(format!("persist reference codebase index: {e}")))?;

        ref_storage
            .mark_status(index.id, CodebaseIndexStatus::Embedding, None)
            .await
            .ok();
        task.update_progress("reference_codebase_embedding".to_string(), 5, 70);

        let code_embed_url = std::env::var("EDGEQUAKE_CODE_EMBEDDING_URL")
            .map_err(|_| TaskError::Process("EDGEQUAKE_CODE_EMBEDDING_URL not set".to_string()))?;
        let code_model = std::env::var("EDGEQUAKE_CODE_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "jina-code-embeddings".to_string());
        let code_dim: usize = std::env::var("EDGEQUAKE_CODE_EMBEDDING_DIMENSION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(896);
        let embedder = JinaEmbedder::new(&code_embed_url, &code_model, code_dim);

        let mut embedded = 0usize;
        for chunk in &build.chunks {
            self.check_cancelled(&cancel_token, "reference_codebase_embedding", &document_id)
                .await?;
            let embedding = embedder
                .embed_code_for_indexing(&chunk.content)
                .await
                .map_err(|e| TaskError::Process(format!("embed codebase chunk: {e}")))?;
            ref_storage
                .insert_embedding(&index, chunk, &code_model, code_dim as i32, &embedding)
                .await
                .map_err(|e| TaskError::Storage(format!("insert codebase embedding: {e}")))?;
            embedded += 1;
        }

        ref_storage
            .mark_status(index.id, CodebaseIndexStatus::Complete, None)
            .await
            .map_err(|e| TaskError::Storage(format!("complete reference codebase index: {e}")))?;
        task.update_progress("completed".to_string(), 5, 100);

        Ok(json!({
            "index_id": index.id,
            "document_id": document_id,
            "document_repo_id": repo_id,
            "repo_commit": snapshot.repo_commit,
            "repo_path": snapshot.repo_path,
            "mode": mode.as_str(),
            "file_count": build.files.iter().filter(|f| f.skipped_reason.is_none()).count(),
            "symbol_count": build.symbols.len(),
            "edge_count": build.edges.len(),
            "chunk_count": build.chunks.len(),
            "embedding_count": embedded,
        }))
    }

    #[cfg(feature = "postgres")]
    async fn reference_codebase_pool(&self) -> TaskResult<sqlx::PgPool> {
        let url = std::env::var("DATABASE_URL")
            .map_err(|_| TaskError::Process("DATABASE_URL not set".to_string()))?;
        sqlx::PgPool::connect(&url)
            .await
            .map_err(|e| TaskError::Storage(format!("connect postgres: {e}")))
    }
}

/// Pull `max_files_override` off the index row. Returns `None` when the
/// column is NULL (explicit POST path) or on any error — the caller
/// should fall back to the global env default, so a transient read
/// failure never blocks the indexer.
#[cfg(feature = "postgres")]
async fn read_max_files_override(
    index_id: uuid::Uuid,
    storage: &edgequake_agents::reference_codebase::PostgresReferenceCodebaseStorage,
) -> Option<i32> {
    let _ = storage;
    let url = std::env::var("DATABASE_URL").ok()?;
    let pool = sqlx::PgPool::connect(&url).await.ok()?;
    sqlx::query_scalar::<_, Option<i32>>(
        "SELECT max_files_override FROM reference_codebase_indexes WHERE id = $1",
    )
    .bind(index_id)
    .fetch_optional(&pool)
    .await
    .ok()
    .flatten()
    .flatten()
}
