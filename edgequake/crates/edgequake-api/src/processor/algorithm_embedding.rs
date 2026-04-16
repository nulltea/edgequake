//! Algorithm embedding task processor.
//!
//! Generates vector embeddings for approved algorithms and stores them
//! in the workspace vector database for semantic retrieval.

use super::*;
use tokio_util::sync::CancellationToken;

impl DocumentTaskProcessor {
    /// Process an algorithm embedding task — generate and store vector embeddings.
    pub(super) async fn process_algorithm_embedding(
        &self,
        task: &mut Task,
        data: edgequake_tasks::AlgorithmEmbeddingData,
        cancel_token: CancellationToken,
    ) -> TaskResult<serde_json::Value> {
        let document_id = &data.document_id;

        info!(
            document_id = %document_id,
            workspace_id = %data.workspace_id,
            algorithm_count = data.algorithm_ids.len(),
            "Processing algorithm embedding task"
        );

        self.check_cancelled(&cancel_token, "algo_embedding", document_id)
            .await?;
        self.update_document_status(document_id, "algo_embedding", None)
            .await
            .ok();
        task.update_progress("algo_embedding".to_string(), 1, 10);

        #[cfg(feature = "postgres")]
        {
            use crate::safety_limits::create_safe_embedding_provider;
            use edgequake_algorithms::{AlgorithmStorage, PostgresAlgorithmStorage};
            use edgequake_storage::traits::WorkspaceVectorConfig;

            let database_url = std::env::var("DATABASE_URL").map_err(|_| {
                TaskError::Process("DATABASE_URL not set".to_string())
            })?;
            let pool = sqlx::PgPool::connect(&database_url).await.map_err(|e| {
                TaskError::Process(format!("Failed to connect to database: {e}"))
            })?;
            let storage = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool));

            let tenant_id = task.tenant_id;
            let workspace_id_uuid = task.workspace_id;

            // Resolve workspace for embedding provider + vector storage
            let ws = self
                .workspace_service
                .as_ref()
                .ok_or_else(|| TaskError::Process("No workspace service".to_string()))?
                .get_workspace(workspace_id_uuid)
                .await
                .map_err(|e| TaskError::Process(format!("Failed to get workspace: {e}")))?
                .ok_or_else(|| TaskError::Process("Workspace not found".to_string()))?;

            let embedding_provider = create_safe_embedding_provider(
                &ws.embedding_provider,
                &ws.embedding_model,
                ws.embedding_dimension,
            )
            .map_err(|e| TaskError::Process(format!("Failed to create embedding provider: {e}")))?;

            let config = WorkspaceVectorConfig {
                workspace_id: workspace_id_uuid,
                dimension: ws.embedding_dimension,
                namespace: "default".to_string(),
            };

            let vector_storage = self
                .vector_registry
                .get_or_create(config)
                .await
                .map_err(|e| TaskError::Process(format!("Failed to get vector storage: {e}")))?;

            let mut embedded_count = 0usize;

            for (i, algo_id_str) in data.algorithm_ids.iter().enumerate() {
                self.check_cancelled(&cancel_token, "algo_embedding", document_id)
                    .await?;

                let algo_uuid = uuid::Uuid::parse_str(algo_id_str).map_err(|e| {
                    TaskError::Process(format!("Invalid algorithm ID {algo_id_str}: {e}"))
                })?;

                let algorithm = match storage
                    .get_algorithm(algo_uuid, tenant_id, workspace_id_uuid)
                    .await
                {
                    Ok(Some(a)) => a,
                    Ok(None) => {
                        warn!(algorithm_id = %algo_id_str, "Algorithm not found, skipping");
                        continue;
                    }
                    Err(e) => {
                        warn!(algorithm_id = %algo_id_str, error = %e, "Failed to fetch algorithm, skipping");
                        continue;
                    }
                };

                // Build text representation for embedding
                let mut text_parts = vec![algorithm.name.clone()];
                if let Some(ref desc) = algorithm.description {
                    text_parts.push(desc.clone());
                }
                for step in &algorithm.steps {
                    text_parts.push(format!(
                        "Step {}: {} {}",
                        step.number, step.action, step.details
                    ));
                }
                if let Some(ref pseudo) = algorithm.pseudocode {
                    text_parts.push(pseudo.clone());
                }
                let embedding_text = text_parts.join("\n");

                // Generate embedding
                let embeddings = embedding_provider
                    .embed(&[embedding_text])
                    .await
                    .map_err(|e| {
                        TaskError::Process(format!("Embedding generation failed: {e}"))
                    })?;

                let embedding = match embeddings.into_iter().next() {
                    Some(e) => e,
                    None => {
                        warn!(algorithm_id = %algo_id_str, "No embedding returned, skipping");
                        continue;
                    }
                };

                let vector_id = format!("algorithm:{}", algorithm.id);
                let metadata = serde_json::json!({
                    "type": "algorithm",
                    "algorithm_id": algorithm.id.to_string(),
                    "document_id": algorithm.document_id,
                    "name": algorithm.name,
                    "description": algorithm.description,
                    "tags": algorithm.tags,
                    "confidence": algorithm.confidence,
                    "tenant_id": algorithm.tenant_id.to_string(),
                    "workspace_id": algorithm.workspace_id.to_string(),
                });

                vector_storage
                    .upsert(&[(vector_id, embedding, metadata)])
                    .await
                    .map_err(|e| {
                        TaskError::Process(format!("Failed to store embedding: {e}"))
                    })?;

                embedded_count += 1;
                let pct = 10 + ((i + 1) * 80 / data.algorithm_ids.len().max(1));
                task.update_progress("algo_embedding".to_string(), 1, pct as u8);
            }

            info!(
                document_id = %document_id,
                embedded = embedded_count,
                total = data.algorithm_ids.len(),
                "Algorithm embedding complete"
            );
        }

        // Restore document to completed status
        self.update_document_status(document_id, "completed", None)
            .await
            .ok();
        task.update_progress("completed".to_string(), 1, 100);

        Ok(json!({
            "document_id": document_id,
            "algorithms_embedded": data.algorithm_ids.len(),
        }))
    }
}
