//! Postgres impl of [`AlgorithmVectorStorage`].
//!
//! Two-stage retrieval:
//!   1. Hit the workspace vector store with metadata filters (tenant + workspace
//!      + document allow-list), ask for `limit * WIDEN_FACTOR` rows, and keep
//!      only those tagged `metadata.type = "algorithm"`.
//!   2. Join the surviving `algorithm_id`s against the `algorithms` table,
//!      filtered by `status = 'approved'`, to pull the structured fields
//!      (`name`, `steps`, `pseudocode`, …) the renderer needs.
//!
//! The widen factor exists because algorithm-type vectors are vastly
//! outnumbered by chunk-type vectors in the same table — a naive
//! `top_k = limit` top-K would be dominated by chunks and return no
//! algorithms at all.

use async_trait::async_trait;
use sqlx::{postgres::PgRow as RawPgRow, FromRow, PgPool, Row as SqlxRow};
use std::collections::HashMap;
use std::sync::Arc;

use uuid::Uuid;

use crate::error::{Result, StorageError};
use crate::traits::{
    AlgorithmSearchHit, AlgorithmStepSummary, AlgorithmVectorStorage, MetadataFilter, VectorStorage,
};

/// Multiplier applied to the caller's `limit` when asking the workspace
/// vector store for candidates. Algorithm-type rows are a small fraction of
/// the total, so we need a wider net.
const WIDEN_FACTOR: usize = 25;
const MIN_CANDIDATES: usize = 200;

pub struct PgAlgorithmVectorStorage {
    pool: Arc<PgPool>,
}

impl PgAlgorithmVectorStorage {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool: Arc::new(pool),
        }
    }
}

#[async_trait]
impl AlgorithmVectorStorage for PgAlgorithmVectorStorage {
    async fn search_approved_algorithms(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        workspace_vectors: &Arc<dyn VectorStorage>,
        query_embedding: &[f32],
        limit: i64,
        max_distance: f64,
        document_ids: Option<&[String]>,
    ) -> Result<Vec<AlgorithmSearchHit>> {
        if limit <= 0 {
            return Ok(Vec::new());
        }

        // vector_type='algorithm' is critical: without it, HNSW's bounded
        // candidate pool is dominated by chunk/entity rows and algorithm
        // rows get evicted before the post-filter even sees them.
        let mf = MetadataFilter {
            document_ids: document_ids.map(|d| d.to_vec()),
            tenant_id: Some(tenant_id.to_string()),
            workspace_id: Some(workspace_id.to_string()),
            vector_type: Some("algorithm".to_string()),
        };

        let top_k = ((limit as usize).saturating_mul(WIDEN_FACTOR)).max(MIN_CANDIDATES);

        let raw = workspace_vectors
            .query_filtered(query_embedding, top_k, None, Some(&mf))
            .await?;

        // Keep only algorithm-type rows and pull out the algorithm UUIDs.
        let mut candidate_scores: HashMap<Uuid, f32> = HashMap::new();
        for r in raw {
            let is_algo = r
                .metadata
                .get("type")
                .and_then(|v| v.as_str())
                .map(|t| t == "algorithm")
                .unwrap_or(false);
            if !is_algo {
                continue;
            }
            let Some(aid) = r
                .metadata
                .get("algorithm_id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok())
            else {
                continue;
            };
            // Distance = 1 - score; caller gave us a max distance threshold.
            let score = r.score;
            let distance = 1.0f32 - score;
            if (distance as f64) > max_distance {
                continue;
            }
            // Keep the best score if dup rows appear.
            candidate_scores
                .entry(aid)
                .and_modify(|prev| {
                    if score > *prev {
                        *prev = score;
                    }
                })
                .or_insert(score);
        }

        if candidate_scores.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<Uuid> = candidate_scores.keys().copied().collect();

        let rows: Vec<Row> = sqlx::query_as::<_, Row>(SQL)
            .bind(&ids)
            .bind(tenant_id)
            .bind(workspace_id)
            .fetch_all(&*self.pool)
            .await
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let mut hits: Vec<AlgorithmSearchHit> = rows
            .into_iter()
            .filter_map(|row| {
                let score = *candidate_scores.get(&row.id)?;
                Some(row.into_hit(score))
            })
            .collect();

        // Sort best-first (lowest distance first) and take `limit`.
        hits.sort_by(|a, b| {
            a.cosine_distance
                .partial_cmp(&b.cosine_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit as usize);

        Ok(hits)
    }
}

const SQL: &str = r#"
    SELECT
        id,
        document_id,
        name,
        description,
        algorithm_type,
        pseudocode,
        complexity,
        steps,
        tags,
        confidence
    FROM algorithms
    WHERE id = ANY($1)
      AND tenant_id = $2
      AND workspace_id = $3
      AND status = 'approved'
"#;

struct Row {
    id: Uuid,
    document_id: String,
    name: String,
    description: Option<String>,
    algorithm_type: String,
    pseudocode: Option<String>,
    complexity: Option<String>,
    steps: serde_json::Value,
    tags: serde_json::Value,
    confidence: String,
}

impl<'r> FromRow<'r, RawPgRow> for Row {
    fn from_row(row: &'r RawPgRow) -> std::result::Result<Self, sqlx::Error> {
        Ok(Self {
            id: row.try_get("id")?,
            document_id: row.try_get("document_id")?,
            name: row.try_get("name")?,
            description: row.try_get("description")?,
            algorithm_type: row
                .try_get::<Option<String>, _>("algorithm_type")?
                .unwrap_or_else(|| "Algorithm".to_string()),
            pseudocode: row.try_get("pseudocode")?,
            complexity: row.try_get("complexity")?,
            steps: row.try_get("steps")?,
            tags: row.try_get("tags")?,
            confidence: row.try_get("confidence")?,
        })
    }
}

impl Row {
    fn into_hit(self, score: f32) -> AlgorithmSearchHit {
        let steps: Vec<AlgorithmStepSummary> = parse_steps(&self.steps);
        let tags: Vec<String> = parse_tags(&self.tags);
        AlgorithmSearchHit {
            algorithm_id: self.id.to_string(),
            document_id: self.document_id,
            name: self.name,
            description: self.description,
            algorithm_type: self.algorithm_type,
            pseudocode: self.pseudocode,
            complexity: self.complexity,
            steps,
            tags,
            confidence: self.confidence,
            cosine_distance: (1.0f32 - score) as f64,
        }
    }
}

fn parse_steps(v: &serde_json::Value) -> Vec<AlgorithmStepSummary> {
    let Some(arr) = v.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|item| {
            let obj = item.as_object()?;
            let number = obj
                .get("number")
                .and_then(|n| n.as_u64())
                .map(|n| n as usize)
                .unwrap_or(0);
            let action = obj
                .get("action")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let details = obj
                .get("details")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            Some(AlgorithmStepSummary {
                number,
                action,
                details,
            })
        })
        .collect()
}

fn parse_tags(v: &serde_json::Value) -> Vec<String> {
    let Some(arr) = v.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|t| t.as_str().map(|s| s.to_string()))
        .collect()
}
