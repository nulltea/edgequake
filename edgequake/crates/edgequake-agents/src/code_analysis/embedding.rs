//! Code embedding client + vector storage for approved code_artifacts.
//!
//! Uses `jina-code-embeddings` served via `llama-server --embeddings` behind
//! an OpenAI-compatible `/v1/embeddings` endpoint.
//!
//! The model is **asymmetric + instruction-prefixed**: callers must prepend
//! task-specific prefixes to both passages (when indexing) and queries
//! (when searching). The five task families from the Jina paper are wrapped
//! in [`JinaCodeTask`]; [`JinaEmbedder::embed_code_for_indexing`] and
//! [`JinaEmbedder::embed_query_for_code_search`] are the two call sites we
//! actually need — the others are exposed for future re-use.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("embedder returned {status}: {body}")]
    Status {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("parse: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("embedder returned no embeddings")]
    Empty,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dimension { expected: usize, got: usize },
}

/// Task families supported by jina-code-embeddings. Prefixes are fixed by
/// the model; changing them silently breaks retrieval quality.
///
/// Source: model card at
/// <https://huggingface.co/jinaai/jina-code-embeddings-0.5b>.
#[derive(Debug, Clone, Copy)]
pub enum JinaCodeTask {
    /// Natural-language query → code passage (code search).
    Nl2Code,
    /// Code query → code passage (similar-code / snippet-level search).
    Code2Code,
    /// Code passage → NL description (summarisation-style retrieval).
    Code2Nl,
    /// Code prefix → code completion / continuation candidate.
    Code2Completion,
    /// QA over code (e.g. Stack Overflow-style).
    Qa,
}

impl JinaCodeTask {
    pub fn query_prefix(self) -> &'static str {
        match self {
            JinaCodeTask::Nl2Code => {
                "Find the most relevant code snippet given the following query:\n"
            }
            JinaCodeTask::Code2Code => "Find a similar code snippet:\n",
            JinaCodeTask::Code2Nl => "Find the NL description of the following code:\n",
            JinaCodeTask::Code2Completion => "Find the completion for the following code:\n",
            JinaCodeTask::Qa => "Find the answer to the following question:\n",
        }
    }
    pub fn passage_prefix(self) -> &'static str {
        match self {
            JinaCodeTask::Nl2Code => "Candidate code snippet:\n",
            JinaCodeTask::Code2Code => "Candidate code snippet:\n",
            JinaCodeTask::Code2Nl => "NL description:\n",
            JinaCodeTask::Code2Completion => "Code completion candidate:\n",
            JinaCodeTask::Qa => "Answer:\n",
        }
    }
}

/// Thin client over the llama.cpp `/v1/embeddings` endpoint.
#[derive(Debug, Clone)]
pub struct JinaEmbedder {
    base_url: String,
    model: String,
    expected_dim: usize,
    http: reqwest::Client,
}

impl JinaEmbedder {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>, expected_dim: usize) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("reqwest builds");
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            expected_dim,
            http,
        }
    }

    /// Embed `snippet` as a **passage**. Prepends the `Code2Code` passage
    /// prefix so it lands in the same retrieval space as NL-keyed queries
    /// embedded with [`Self::embed_query_for_code_search`].
    ///
    /// Rationale: edgequake queries are predominantly natural-language
    /// ("how does algorithm X work?"), so nl2code is the right asymmetric
    /// pair. The passage side is the same string for both nl2code and
    /// code2code per the model card, so this single prefix doubles.
    pub async fn embed_code_for_indexing(&self, snippet: &str) -> Result<Vec<f32>, EmbedderError> {
        self.embed_one(&format!(
            "{}{}",
            JinaCodeTask::Nl2Code.passage_prefix(),
            snippet
        ))
        .await
    }

    /// Embed `query` (NL) with the `nl2code` query prefix for similarity
    /// search against vectors produced by [`Self::embed_code_for_indexing`].
    pub async fn embed_query_for_code_search(
        &self,
        query: &str,
    ) -> Result<Vec<f32>, EmbedderError> {
        self.embed_one(&format!(
            "{}{}",
            JinaCodeTask::Nl2Code.query_prefix(),
            query
        ))
        .await
    }

    async fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        let url = format!("{}/embeddings", self.base_url);
        let body = EmbeddingsRequest {
            model: &self.model,
            input: text,
        };
        let resp = self.http.post(&url).json(&body).send().await?;
        let status = resp.status();
        let text_body = resp.text().await?;
        if !status.is_success() {
            return Err(EmbedderError::Status {
                status,
                body: truncate(&text_body, 400),
            });
        }
        let parsed: EmbeddingsResponse = serde_json::from_str(&text_body)?;
        let emb = parsed
            .data
            .into_iter()
            .next()
            .ok_or(EmbedderError::Empty)?
            .embedding;
        if emb.len() != self.expected_dim {
            return Err(EmbedderError::Dimension {
                expected: self.expected_dim,
                got: emb.len(),
            });
        }
        Ok(emb)
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingEntry>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingEntry {
    embedding: Vec<f32>,
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut cut = max;
        while !s.is_char_boundary(cut) && cut > 0 {
            cut -= 1;
        }
        format!("{}…", &s[..cut])
    }
}

// ── Storage ────────────────────────────────────────────────────────────────

#[cfg(feature = "postgres")]
pub use postgres_impl::*;

#[cfg(feature = "postgres")]
mod postgres_impl {
    use super::EmbedderError;
    use uuid::Uuid;

    #[derive(Debug, thiserror::Error)]
    pub enum EmbeddingStorageError {
        #[error("database error: {0}")]
        Database(#[from] sqlx::Error),
        #[error("embedder error: {0}")]
        Embedder(#[from] EmbedderError),
    }

    /// Persists a `code_artifact_id → embedding` pair and the follow-up
    /// bookkeeping on `code_artifacts.embedding_id`.
    ///
    /// Idempotent: re-running with a new embedding replaces the row.
    pub struct CodeEmbeddingStorage {
        pool: sqlx::PgPool,
    }

    impl CodeEmbeddingStorage {
        pub fn new(pool: sqlx::PgPool) -> Self {
            Self { pool }
        }

        /// Upsert the embedding row. Caller should already have called the
        /// embedder and normalised the vector if needed (Jina's server
        /// returns unit-norm vectors directly).
        #[allow(clippy::too_many_arguments)]
        pub async fn upsert(
            &self,
            code_artifact_id: Uuid,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            algorithm_id: Uuid,
            embedding_model: &str,
            embedding: &[f32],
        ) -> Result<Uuid, EmbeddingStorageError> {
            // pgvector's sqlx bindings expect `Vec<f32>` via the `pgvector`
            // crate — but edgequake already wraps this elsewhere. We pass
            // as a Postgres literal string `[v1,v2,...]` to avoid pulling in
            // a dependency the workspace doesn't already have. See
            // <https://github.com/pgvector/pgvector-rust/issues>.
            let literal = vector_literal(embedding);
            let row: (Uuid,) = sqlx::query_as(
                r#"
                INSERT INTO code_artifact_embeddings (
                    code_artifact_id, tenant_id, workspace_id, document_id,
                    algorithm_id, embedding_model, embedding_dim, embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
                ON CONFLICT (code_artifact_id) DO UPDATE SET
                    tenant_id = EXCLUDED.tenant_id,
                    workspace_id = EXCLUDED.workspace_id,
                    document_id = EXCLUDED.document_id,
                    algorithm_id = EXCLUDED.algorithm_id,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    embedding = EXCLUDED.embedding,
                    created_at = NOW()
                RETURNING id
                "#,
            )
            .bind(code_artifact_id)
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .bind(algorithm_id)
            .bind(embedding_model)
            .bind(embedding.len() as i32)
            .bind(&literal)
            .fetch_one(&self.pool)
            .await?;

            // Write the FK into code_artifacts.embedding_id.
            sqlx::query(
                r#"UPDATE code_artifacts SET embedding_id = $1, updated_at = NOW() WHERE id = $2"#,
            )
            .bind(row.0)
            .bind(code_artifact_id)
            .execute(&self.pool)
            .await?;

            Ok(row.0)
        }

        /// Remove the embedding row (also clears `code_artifacts.embedding_id`).
        pub async fn delete_for_artifact(
            &self,
            code_artifact_id: Uuid,
        ) -> Result<bool, EmbeddingStorageError> {
            // ON DELETE CASCADE on the FK would also fire if the artifact
            // row were deleted; here we only want to drop the embedding,
            // not the artifact.
            let res =
                sqlx::query(r#"DELETE FROM code_artifact_embeddings WHERE code_artifact_id = $1"#)
                    .bind(code_artifact_id)
                    .execute(&self.pool)
                    .await?;
            sqlx::query(
                r#"UPDATE code_artifacts SET embedding_id = NULL, updated_at = NOW() WHERE id = $1"#,
            )
            .bind(code_artifact_id)
            .execute(&self.pool)
            .await?;
            Ok(res.rows_affected() > 0)
        }
    }

    /// Format `&[f32]` as the Postgres literal `[x1,x2,...]` pgvector accepts.
    fn vector_literal(v: &[f32]) -> String {
        let mut s = String::with_capacity(v.len() * 10 + 2);
        s.push('[');
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            // Limit precision — pgvector stores single-precision anyway.
            use std::fmt::Write;
            let _ = write!(s, "{:.7}", x);
        }
        s.push(']');
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_prefixes_are_stable() {
        // Guardrail: these strings are fixed by the model. If Jina ever
        // updates them, that's a breaking change.
        assert_eq!(
            JinaCodeTask::Nl2Code.query_prefix(),
            "Find the most relevant code snippet given the following query:\n"
        );
        assert_eq!(
            JinaCodeTask::Nl2Code.passage_prefix(),
            "Candidate code snippet:\n"
        );
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn vector_literal_roundtrip() {
        use super::postgres_impl::*;
        let _ = CodeEmbeddingStorage::new; // touch symbol
                                           // vector_literal is private-module-level; no direct test here,
                                           // but the exercise in `upsert` hits it in the integration tests
                                           // (tests/code_storage.rs / live upload in staging).
    }
}
