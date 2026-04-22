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

    /// Search approved algorithms across documents in a workspace.
    async fn search_algorithms(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        query: Option<&str>,
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

/// Tokenize algorithm search text for deterministic lexical matching.
///
/// This intentionally stays lighter than BM25: no corpus statistics, no RRF,
/// just normalized terms that can boost exact algorithm names and paper terms.
pub fn tokenize_algorithm_query(query: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "of",
        "on", "or", "the", "to", "using", "with",
    ];

    let mut tokens = Vec::new();
    for token in query
        .split(|c: char| !c.is_alphanumeric())
        .map(|s| s.trim().to_lowercase())
        .filter(|s| s.len() >= 2 && !STOPWORDS.contains(&s.as_str()))
    {
        if !tokens.contains(&token) {
            tokens.push(token);
        }
    }
    tokens
}

/// Score a single approved algorithm against tokenized query terms.
///
/// Higher weights are assigned to algorithm identity fields. Longer bodies
/// still contribute, but cannot swamp exact title/tag matches.
pub fn score_algorithm_match(algorithm: &Algorithm, tokens: &[String], raw_query: &str) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }

    let raw_query = raw_query.trim().to_lowercase();
    let name = algorithm.name.to_lowercase();
    let algorithm_type = algorithm.algorithm_type.to_lowercase();
    let description = algorithm
        .description
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    let complexity = algorithm
        .complexity
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    let pseudocode = algorithm
        .pseudocode
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    let tags = algorithm.tags.join(" ").to_lowercase();
    let steps = algorithm
        .steps
        .iter()
        .map(|s| format!("{} {}", s.action, s.details))
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let io = algorithm
        .inputs
        .iter()
        .chain(algorithm.outputs.iter())
        .map(|v| format!("{} {}", v.name, v.description))
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();

    let mut score = 0.0;
    if !raw_query.is_empty() && name.contains(&raw_query) {
        score += 20.0;
    }

    for token in tokens {
        if name == *token {
            score += 12.0;
        } else if name.contains(token) {
            score += 8.0;
        }
        if tags.split_whitespace().any(|tag| tag == token) {
            score += 6.0;
        } else if tags.contains(token) {
            score += 4.0;
        }
        if algorithm_type.contains(token) {
            score += 3.0;
        }
        if description.contains(token) {
            score += 3.0;
        }
        if complexity.contains(token) {
            score += 2.0;
        }
        if pseudocode.contains(token) {
            score += 2.0;
        }
        if steps.contains(token) {
            score += 1.5;
        }
        if io.contains(token) {
            score += 1.0;
        }
    }

    score
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
        document_id: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<(Vec<Algorithm>, i64), AlgorithmStorageError> {
        if query.map(|q| q.trim().is_empty()).unwrap_or(true) {
            let (rows, total) = if let Some(document_id) = document_id {
                let total = sqlx::query_scalar::<_, i64>(
                    r#"
                    SELECT COUNT(*)::BIGINT FROM algorithms
                    WHERE tenant_id = $1
                      AND workspace_id = $2
                      AND status = 'approved'
                      AND document_id = $3
                    "#,
                )
                .bind(tenant_id)
                .bind(workspace_id)
                .bind(document_id)
                .fetch_one(self.pool.as_ref())
                .await?;

                let rows = sqlx::query_as::<_, AlgorithmRow>(
                    r#"
                    SELECT * FROM algorithms
                    WHERE tenant_id = $1
                      AND workspace_id = $2
                      AND status = 'approved'
                      AND document_id = $3
                    ORDER BY created_at DESC
                    LIMIT $4 OFFSET $5
                    "#,
                )
                .bind(tenant_id)
                .bind(workspace_id)
                .bind(document_id)
                .bind(limit.max(0))
                .bind(offset.max(0))
                .fetch_all(self.pool.as_ref())
                .await?;

                (rows, total)
            } else {
                let total = sqlx::query_scalar::<_, i64>(
                    r#"
                    SELECT COUNT(*)::BIGINT FROM algorithms
                    WHERE tenant_id = $1
                      AND workspace_id = $2
                      AND status = 'approved'
                    "#,
                )
                .bind(tenant_id)
                .bind(workspace_id)
                .fetch_one(self.pool.as_ref())
                .await?;

                let rows = sqlx::query_as::<_, AlgorithmRow>(
                    r#"
                    SELECT * FROM algorithms
                    WHERE tenant_id = $1
                      AND workspace_id = $2
                      AND status = 'approved'
                    ORDER BY created_at DESC
                    LIMIT $3 OFFSET $4
                    "#,
                )
                .bind(tenant_id)
                .bind(workspace_id)
                .bind(limit.max(0))
                .bind(offset.max(0))
                .fetch_all(self.pool.as_ref())
                .await?;

                (rows, total)
            };

            let algorithms = rows
                .into_iter()
                .map(|r| r.into_algorithm())
                .collect::<Result<Vec<_>, _>>()?;

            return Ok((algorithms, total));
        }

        let rows = if let Some(document_id) = document_id {
            sqlx::query_as::<_, AlgorithmRow>(
                r#"
                SELECT * FROM algorithms
                WHERE tenant_id = $1
                  AND workspace_id = $2
                  AND status = 'approved'
                  AND document_id = $3
                ORDER BY created_at DESC
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .fetch_all(self.pool.as_ref())
            .await?
        } else {
            sqlx::query_as::<_, AlgorithmRow>(
                r#"
                SELECT * FROM algorithms
                WHERE tenant_id = $1
                  AND workspace_id = $2
                  AND status = 'approved'
                ORDER BY created_at DESC
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .fetch_all(self.pool.as_ref())
            .await?
        };

        let mut scored = rows
            .into_iter()
            .map(|r| r.into_algorithm())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|algorithm| {
                let score = query
                    .map(|q| score_algorithm_match(&algorithm, &tokenize_algorithm_query(q), q))
                    .unwrap_or(0.0);
                (algorithm, score)
            })
            .collect::<Vec<_>>();

        if query.map(|q| !q.trim().is_empty()).unwrap_or(false) {
            scored.retain(|(_, score)| *score > 0.0);
            scored.sort_by(|(a_algo, a_score), (b_algo, b_score)| {
                b_score
                    .partial_cmp(a_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b_algo.created_at.cmp(&a_algo.created_at))
            });
        }

        let total = scored.len() as i64;
        let start = offset.max(0) as usize;
        let end = start.saturating_add(limit.max(0) as usize);
        let algorithms = scored
            .into_iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .map(|(algorithm, _)| algorithm)
            .collect();

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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_algorithm() -> Algorithm {
        Algorithm {
            id: Uuid::new_v4(),
            tenant_id: Uuid::new_v4(),
            workspace_id: Uuid::new_v4(),
            document_id: Uuid::new_v4().to_string(),
            name: "SAP".to_string(),
            algorithm_type: "Scheme".to_string(),
            description: Some(
                "Scale and Perturb approximate distance comparison preserving symmetric encryption"
                    .to_string(),
            ),
            steps: vec![AlgorithmStep {
                number: 1,
                action: "Scale plaintext distance".to_string(),
                details: "Perturb the scaled value before encryption".to_string(),
                math: None,
            }],
            inputs: vec![AlgorithmIO {
                name: "plaintext".to_string(),
                io_type: "input".to_string(),
                description: "numeric vector".to_string(),
            }],
            outputs: Vec::new(),
            preconditions: Vec::new(),
            complexity: Some("linear".to_string()),
            mathematical_notation: None,
            pseudocode: Some("scaled <- scale(x); ciphertext <- encrypt(scaled)".to_string()),
            tags: vec!["encryption".to_string(), "distance".to_string()],
            confidence: "high".to_string(),
            status: AlgorithmStatus::Approved,
            verification_status: None,
            verification_details: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn tokenized_scoring_matches_long_algorithm_queries() {
        let query =
            "SAP Scale and Perturb approximate distance comparison preserving symmetric encryption";
        let tokens = tokenize_algorithm_query(query);

        assert!(tokens.contains(&"sap".to_string()));
        assert!(!tokens.contains(&"and".to_string()));
        assert!(score_algorithm_match(&test_algorithm(), &tokens, query) > 20.0);
    }
}
