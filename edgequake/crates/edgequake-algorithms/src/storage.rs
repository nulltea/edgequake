//! PostgreSQL storage for extracted algorithms.
//!
//! Follows EdgeQuake's multi-tenant pattern: all queries filter by tenant_id AND workspace_id.

use chrono::Utc;
use sqlx::PgPool;
use std::sync::Arc;
use uuid::Uuid;

use crate::types::{Algorithm, AlgorithmIO, AlgorithmStatus, AlgorithmStep};

/// Trait for algorithm storage operations.
#[async_trait::async_trait]
pub trait AlgorithmStorage: Send + Sync {
    /// Store a batch of extracted algorithms.
    async fn create_algorithms(
        &self,
        algorithms: &[Algorithm],
    ) -> Result<(), AlgorithmStorageError>;

    /// List algorithms for a document, optionally filtered by status.
    async fn list_algorithms(
        &self,
        document_id: &str,
        tenant_id: Uuid,
        workspace_id: Uuid,
        status: Option<AlgorithmStatus>,
    ) -> Result<Vec<Algorithm>, AlgorithmStorageError>;

    /// Update the review status of a single algorithm.
    async fn update_algorithm_status(
        &self,
        id: Uuid,
        tenant_id: Uuid,
        workspace_id: Uuid,
        status: AlgorithmStatus,
    ) -> Result<(), AlgorithmStorageError>;

    /// Get a single algorithm by ID.
    async fn get_algorithm(
        &self,
        id: Uuid,
        tenant_id: Uuid,
        workspace_id: Uuid,
    ) -> Result<Option<Algorithm>, AlgorithmStorageError>;

    /// Delete a single algorithm by ID.
    async fn delete_algorithm(
        &self,
        id: Uuid,
        tenant_id: Uuid,
        workspace_id: Uuid,
    ) -> Result<bool, AlgorithmStorageError>;

    /// Delete all algorithms for a document.
    async fn delete_algorithms_by_document(
        &self,
        document_id: &str,
        tenant_id: Uuid,
        workspace_id: Uuid,
    ) -> Result<u64, AlgorithmStorageError>;

    /// Search algorithms across documents in a workspace.
    async fn search_algorithms(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        query: Option<&str>,
        status: Option<AlgorithmStatus>,
        document_id: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<(Vec<Algorithm>, i64), AlgorithmStorageError>;

    /// Count algorithms grouped by document for a workspace.
    ///
    /// Returns a `(document_id, count)` list. Used by the document-list UI
    /// to decide whether to show the per-row "Algorithms" action button —
    /// fetching one row per doc via `list_algorithms` on N rows is wasteful,
    /// so we batch.
    async fn counts_by_document(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
    ) -> Result<Vec<(String, i64)>, AlgorithmStorageError>;
}

#[derive(Debug, thiserror::Error)]
pub enum AlgorithmStorageError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// PostgreSQL-backed algorithm storage.
pub struct PostgresAlgorithmStorage {
    pool: Arc<PgPool>,
}

impl PostgresAlgorithmStorage {
    pub fn new(pool: Arc<PgPool>) -> Self {
        Self { pool }
    }
}

#[async_trait::async_trait]
impl AlgorithmStorage for PostgresAlgorithmStorage {
    async fn create_algorithms(
        &self,
        algorithms: &[Algorithm],
    ) -> Result<(), AlgorithmStorageError> {
        for algo in algorithms {
            let steps_json = serde_json::to_value(&algo.steps)
                .map_err(|e| AlgorithmStorageError::Serialization(e.to_string()))?;
            let inputs_json = serde_json::to_value(&algo.inputs)
                .map_err(|e| AlgorithmStorageError::Serialization(e.to_string()))?;
            let outputs_json = serde_json::to_value(&algo.outputs)
                .map_err(|e| AlgorithmStorageError::Serialization(e.to_string()))?;
            let preconditions_json = serde_json::to_value(&algo.preconditions)
                .map_err(|e| AlgorithmStorageError::Serialization(e.to_string()))?;
            let tags_json = serde_json::to_value(&algo.tags)
                .map_err(|e| AlgorithmStorageError::Serialization(e.to_string()))?;

            sqlx::query(
                r#"
                INSERT INTO algorithms (
                    id, tenant_id, workspace_id, document_id, name, algorithm_type,
                    description, steps, inputs, outputs, preconditions, complexity,
                    mathematical_notation, pseudocode, tags, confidence, status,
                    verification_status, verification_details, created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $9, $10, $11, $12,
                    $13, $14, $15, $16, $17,
                    $18, $19, $20, $21
                )
                "#,
            )
            .bind(algo.id)
            .bind(algo.tenant_id)
            .bind(algo.workspace_id)
            .bind(&algo.document_id)
            .bind(&algo.name)
            .bind(&algo.algorithm_type)
            .bind(&algo.description)
            .bind(&steps_json)
            .bind(&inputs_json)
            .bind(&outputs_json)
            .bind(&preconditions_json)
            .bind(&algo.complexity)
            .bind(&algo.mathematical_notation)
            .bind(&algo.pseudocode)
            .bind(&tags_json)
            .bind(&algo.confidence)
            .bind(algo.status.to_string())
            .bind(&algo.verification_status)
            .bind(&algo.verification_details)
            .bind(algo.created_at)
            .bind(algo.updated_at)
            .execute(self.pool.as_ref())
            .await?;
        }
        Ok(())
    }

    async fn list_algorithms(
        &self,
        document_id: &str,
        tenant_id: Uuid,
        workspace_id: Uuid,
        status: Option<AlgorithmStatus>,
    ) -> Result<Vec<Algorithm>, AlgorithmStorageError> {
        let rows = if let Some(status) = status {
            sqlx::query_as::<_, AlgorithmRow>(
                r#"
                SELECT * FROM algorithms
                WHERE document_id = $1 AND tenant_id = $2 AND workspace_id = $3 AND status = $4
                ORDER BY created_at ASC
                "#,
            )
            .bind(document_id)
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(status.to_string())
            .fetch_all(self.pool.as_ref())
            .await?
        } else {
            sqlx::query_as::<_, AlgorithmRow>(
                r#"
                SELECT * FROM algorithms
                WHERE document_id = $1 AND tenant_id = $2 AND workspace_id = $3
                ORDER BY created_at ASC
                "#,
            )
            .bind(document_id)
            .bind(tenant_id)
            .bind(workspace_id)
            .fetch_all(self.pool.as_ref())
            .await?
        };

        rows.into_iter()
            .map(|r| r.into_algorithm())
            .collect::<Result<Vec<_>, _>>()
    }

    async fn update_algorithm_status(
        &self,
        id: Uuid,
        tenant_id: Uuid,
        workspace_id: Uuid,
        status: AlgorithmStatus,
    ) -> Result<(), AlgorithmStorageError> {
        sqlx::query(
            r#"
            UPDATE algorithms
            SET status = $1, updated_at = $2
            WHERE id = $3 AND tenant_id = $4 AND workspace_id = $5
            "#,
        )
        .bind(status.to_string())
        .bind(Utc::now())
        .bind(id)
        .bind(tenant_id)
        .bind(workspace_id)
        .execute(self.pool.as_ref())
        .await?;
        Ok(())
    }

    async fn get_algorithm(
        &self,
        id: Uuid,
        tenant_id: Uuid,
        workspace_id: Uuid,
    ) -> Result<Option<Algorithm>, AlgorithmStorageError> {
        let row = sqlx::query_as::<_, AlgorithmRow>(
            r#"
            SELECT * FROM algorithms
            WHERE id = $1 AND tenant_id = $2 AND workspace_id = $3
            "#,
        )
        .bind(id)
        .bind(tenant_id)
        .bind(workspace_id)
        .fetch_optional(self.pool.as_ref())
        .await?;

        match row {
            Some(r) => Ok(Some(r.into_algorithm()?)),
            None => Ok(None),
        }
    }

    async fn delete_algorithm(
        &self,
        id: Uuid,
        tenant_id: Uuid,
        workspace_id: Uuid,
    ) -> Result<bool, AlgorithmStorageError> {
        let result = sqlx::query(
            r#"
            DELETE FROM algorithms
            WHERE id = $1 AND tenant_id = $2 AND workspace_id = $3
            "#,
        )
        .bind(id)
        .bind(tenant_id)
        .bind(workspace_id)
        .execute(self.pool.as_ref())
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn delete_algorithms_by_document(
        &self,
        document_id: &str,
        tenant_id: Uuid,
        workspace_id: Uuid,
    ) -> Result<u64, AlgorithmStorageError> {
        let result = sqlx::query(
            r#"
            DELETE FROM algorithms
            WHERE document_id = $1 AND tenant_id = $2 AND workspace_id = $3
            "#,
        )
        .bind(document_id)
        .bind(tenant_id)
        .bind(workspace_id)
        .execute(self.pool.as_ref())
        .await?;
        Ok(result.rows_affected())
    }

    async fn search_algorithms(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        query: Option<&str>,
        status: Option<AlgorithmStatus>,
        document_id: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<(Vec<Algorithm>, i64), AlgorithmStorageError> {
        // Build dynamic WHERE clause
        let mut conditions = vec![
            "tenant_id = $1".to_string(),
            "workspace_id = $2".to_string(),
        ];
        let mut param_idx = 3;

        if query.is_some() {
            conditions.push(format!(
                "(name ILIKE '%' || ${p} || '%' OR description ILIKE '%' || ${p} || '%')",
                p = param_idx
            ));
            param_idx += 1;
        }
        if status.is_some() {
            conditions.push(format!("status = ${param_idx}"));
            param_idx += 1;
        }
        if document_id.is_some() {
            conditions.push(format!("document_id = ${param_idx}"));
            // param_idx += 1; // unused after this
        }

        let where_clause = conditions.join(" AND ");
        let count_sql = format!("SELECT COUNT(*) as count FROM algorithms WHERE {where_clause}");
        let data_sql = format!(
            "SELECT * FROM algorithms WHERE {where_clause} ORDER BY created_at DESC LIMIT {limit} OFFSET {offset}"
        );

        // We need to use raw queries with dynamic binding.
        // For simplicity, use sqlx::query_scalar and sqlx::query_as with manual binding.
        // Since sqlx doesn't support dynamic parameter counts easily, we build the queries
        // with all possible parameters and use conditional binding.

        // Count query
        let mut count_query = sqlx::query_scalar::<_, i64>(&count_sql);
        count_query = count_query.bind(tenant_id).bind(workspace_id);
        if let Some(q) = query {
            count_query = count_query.bind(q);
        }
        if let Some(s) = status {
            count_query = count_query.bind(s.to_string());
        }
        if let Some(d) = document_id {
            count_query = count_query.bind(d);
        }
        let total = count_query.fetch_one(self.pool.as_ref()).await?;

        // Data query
        let mut data_query = sqlx::query_as::<_, AlgorithmRow>(&data_sql);
        data_query = data_query.bind(tenant_id).bind(workspace_id);
        if let Some(q) = query {
            data_query = data_query.bind(q);
        }
        if let Some(s) = status {
            data_query = data_query.bind(s.to_string());
        }
        if let Some(d) = document_id {
            data_query = data_query.bind(d);
        }
        let rows = data_query.fetch_all(self.pool.as_ref()).await?;

        let algorithms = rows
            .into_iter()
            .map(|r| r.into_algorithm())
            .collect::<Result<Vec<_>, _>>()?;

        Ok((algorithms, total))
    }

    async fn counts_by_document(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
    ) -> Result<Vec<(String, i64)>, AlgorithmStorageError> {
        let rows: Vec<(String, i64)> = sqlx::query_as(
            r#"
            SELECT document_id, COUNT(*)::BIGINT
            FROM algorithms
            WHERE tenant_id = $1 AND workspace_id = $2
            GROUP BY document_id
            "#,
        )
        .bind(tenant_id)
        .bind(workspace_id)
        .fetch_all(self.pool.as_ref())
        .await?;
        Ok(rows)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//                         DATABASE ROW MAPPING
// ─────────────────────────────────────────────────────────────────────────────

/// Raw database row — JSONB columns are serde_json::Value.
#[derive(Debug, sqlx::FromRow)]
struct AlgorithmRow {
    id: Uuid,
    tenant_id: Uuid,
    workspace_id: Uuid,
    document_id: String,
    name: String,
    #[sqlx(default)]
    algorithm_type: Option<String>,
    description: Option<String>,
    steps: serde_json::Value,
    inputs: serde_json::Value,
    outputs: serde_json::Value,
    preconditions: serde_json::Value,
    complexity: Option<String>,
    mathematical_notation: Option<String>,
    pseudocode: Option<String>,
    tags: serde_json::Value,
    confidence: String,
    status: String,
    verification_status: Option<String>,
    verification_details: Option<serde_json::Value>,
    created_at: chrono::DateTime<Utc>,
    updated_at: chrono::DateTime<Utc>,
}

impl AlgorithmRow {
    fn into_algorithm(self) -> Result<Algorithm, AlgorithmStorageError> {
        let steps: Vec<AlgorithmStep> = serde_json::from_value(self.steps)
            .map_err(|e| AlgorithmStorageError::Serialization(format!("steps: {e}")))?;
        let inputs: Vec<AlgorithmIO> = serde_json::from_value(self.inputs)
            .map_err(|e| AlgorithmStorageError::Serialization(format!("inputs: {e}")))?;
        let outputs: Vec<AlgorithmIO> = serde_json::from_value(self.outputs)
            .map_err(|e| AlgorithmStorageError::Serialization(format!("outputs: {e}")))?;
        let preconditions: Vec<String> = serde_json::from_value(self.preconditions)
            .map_err(|e| AlgorithmStorageError::Serialization(format!("preconditions: {e}")))?;
        let tags: Vec<String> = serde_json::from_value(self.tags)
            .map_err(|e| AlgorithmStorageError::Serialization(format!("tags: {e}")))?;
        let status: AlgorithmStatus = self
            .status
            .parse()
            .map_err(|e: String| AlgorithmStorageError::Serialization(e))?;

        Ok(Algorithm {
            id: self.id,
            tenant_id: self.tenant_id,
            workspace_id: self.workspace_id,
            document_id: self.document_id,
            name: self.name,
            algorithm_type: self
                .algorithm_type
                .unwrap_or_else(|| "Algorithm".to_string()),
            description: self.description,
            steps,
            inputs,
            outputs,
            preconditions,
            complexity: self.complexity,
            mathematical_notation: self.mathematical_notation,
            pseudocode: self.pseudocode,
            tags,
            confidence: self.confidence,
            status,
            verification_status: self.verification_status,
            verification_details: self.verification_details,
            created_at: self.created_at,
            updated_at: self.updated_at,
        })
    }
}
