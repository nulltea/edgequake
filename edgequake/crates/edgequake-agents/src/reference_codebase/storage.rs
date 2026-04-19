use async_trait::async_trait;
use uuid::Uuid;

use super::types::{
    CodebaseChunk, CodebaseEdge, CodebaseFile, CodebaseIndex, CodebaseIndexMode,
    CodebaseIndexStatus, CodebaseSubgraph, CodebaseSymbol, IndexBuildOutput, SubgraphEdge,
    SubgraphNode,
};

#[derive(Debug, thiserror::Error)]
pub enum ReferenceCodebaseStorageError {
    #[cfg(feature = "postgres")]
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("decode error: {0}")]
    Decode(String),
}

#[async_trait]
pub trait ReferenceCodebaseStorage: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    async fn start_index(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        document_repo_id: Uuid,
        repo_url: &str,
        repo_commit: &str,
        repo_path: &str,
        repo_license: Option<&str>,
        mode: CodebaseIndexMode,
        force_reindex: bool,
    ) -> Result<CodebaseIndex, ReferenceCodebaseStorageError>;

    async fn mark_status(
        &self,
        index_id: Uuid,
        status: CodebaseIndexStatus,
        error_message: Option<&str>,
    ) -> Result<(), ReferenceCodebaseStorageError>;

    async fn replace_index_data(
        &self,
        index: &CodebaseIndex,
        output: &IndexBuildOutput,
    ) -> Result<(), ReferenceCodebaseStorageError>;

    async fn insert_embedding(
        &self,
        index: &CodebaseIndex,
        chunk: &CodebaseChunk,
        embedding_model: &str,
        embedding_dim: i32,
        embedding: &[f32],
    ) -> Result<(), ReferenceCodebaseStorageError>;

    async fn get_index(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        index_id: Uuid,
    ) -> Result<Option<CodebaseIndex>, ReferenceCodebaseStorageError>;

    async fn list_indexes_for_repo(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_repo_id: Uuid,
    ) -> Result<Vec<CodebaseIndex>, ReferenceCodebaseStorageError>;

    /// Resolve `code_artifact_id → symbol_id` within an index. Finds the
    /// `reference_codebase_symbols` rows whose file + line-range overlap
    /// the code artifact's span. Used as the seed-set resolver when a
    /// caller passes `anchor_artifact_id` to the graph endpoint.
    ///
    /// Returns an empty list when the artifact has no matching symbol
    /// in the index (e.g. the indexer couldn't parse the file).
    async fn anchor_symbols_for_artifact(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        index_id: Uuid,
        code_artifact_id: Uuid,
    ) -> Result<Vec<Uuid>, ReferenceCodebaseStorageError>;

    /// Resolve a free-text symbol name within an index. Exact-match
    /// first, then case-insensitive fallback. Used by the MCP
    /// `get_symbol_neighborhood` flow and by the `mode='full'` UI
    /// fallback when no anchor is selected.
    async fn symbol_ids_by_name(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        index_id: Uuid,
        symbol_name: &str,
        limit: i64,
    ) -> Result<Vec<Uuid>, ReferenceCodebaseStorageError>;

    /// Return the top-N symbols by symbol-level degree (incoming +
    /// outgoing edges). Used as the fallback seed set when the graph
    /// endpoint is called on a `mode='full'` index with no anchor.
    async fn top_symbols_by_degree(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        index_id: Uuid,
        limit: i64,
    ) -> Result<Vec<Uuid>, ReferenceCodebaseStorageError>;

    /// BFS the symbol-edge graph starting from `seed_symbol_ids`, out
    /// `hops` levels, bounded by `max_nodes`. Returns nodes (enriched
    /// with their best-matching chunk) and the edges between them.
    ///
    /// Only symbol↔symbol edges are returned — file-only edges have no
    /// renderable node and would clutter the client's graph layout.
    #[allow(clippy::too_many_arguments)]
    async fn fetch_subgraph(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        index_id: Uuid,
        seed_symbol_ids: &[Uuid],
        hops: i32,
        max_nodes: i32,
    ) -> Result<CodebaseSubgraph, ReferenceCodebaseStorageError>;
}

#[cfg(feature = "postgres")]
pub use postgres::PostgresReferenceCodebaseStorage;

#[cfg(feature = "postgres")]
mod postgres {
    use super::*;
    use chrono::{DateTime, Utc};
    use sqlx::{postgres::PgRow, PgPool, Row};

    pub struct PostgresReferenceCodebaseStorage {
        pool: PgPool,
    }

    impl PostgresReferenceCodebaseStorage {
        pub fn new(pool: PgPool) -> Self {
            Self { pool }
        }
    }

    #[async_trait]
    impl ReferenceCodebaseStorage for PostgresReferenceCodebaseStorage {
        async fn start_index(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_id: &str,
            document_repo_id: Uuid,
            repo_url: &str,
            repo_commit: &str,
            repo_path: &str,
            repo_license: Option<&str>,
            mode: CodebaseIndexMode,
            force_reindex: bool,
        ) -> Result<CodebaseIndex, ReferenceCodebaseStorageError> {
            if force_reindex {
                sqlx::query(
                    r#"
                    DELETE FROM reference_codebase_indexes
                    WHERE tenant_id = $1 AND workspace_id = $2
                      AND document_repo_id = $3 AND repo_commit = $4 AND mode = $5
                    "#,
                )
                .bind(tenant_id)
                .bind(workspace_id)
                .bind(document_repo_id)
                .bind(repo_commit)
                .bind(mode.as_str())
                .execute(&self.pool)
                .await?;
            }

            let row = sqlx::query(
                r#"
                INSERT INTO reference_codebase_indexes (
                    tenant_id, workspace_id, document_id, document_repo_id,
                    repo_url, repo_commit, repo_path, repo_license,
                    mode, status, started_at, updated_at
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'queued',NOW(),NOW())
                ON CONFLICT (tenant_id, workspace_id, document_repo_id, repo_commit, mode)
                DO UPDATE SET
                    repo_path = EXCLUDED.repo_path,
                    repo_license = EXCLUDED.repo_license,
                    status = 'queued',
                    error_message = NULL,
                    started_at = NOW(),
                    completed_at = NULL,
                    updated_at = NOW()
                RETURNING *
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .bind(document_repo_id)
            .bind(repo_url)
            .bind(repo_commit)
            .bind(repo_path)
            .bind(repo_license)
            .bind(mode.as_str())
            .fetch_one(&self.pool)
            .await?;
            decode_index(row)
        }

        async fn mark_status(
            &self,
            index_id: Uuid,
            status: CodebaseIndexStatus,
            error_message: Option<&str>,
        ) -> Result<(), ReferenceCodebaseStorageError> {
            let complete = status == CodebaseIndexStatus::Complete;
            let failed = status == CodebaseIndexStatus::Failed;
            sqlx::query(
                r#"
                UPDATE reference_codebase_indexes
                SET status = $2,
                    error_message = CASE WHEN $4 THEN $3 ELSE NULL END,
                    completed_at = CASE WHEN $5 OR $4 THEN NOW() ELSE completed_at END,
                    updated_at = NOW()
                WHERE id = $1
                "#,
            )
            .bind(index_id)
            .bind(status.as_str())
            .bind(error_message)
            .bind(failed)
            .bind(complete)
            .execute(&self.pool)
            .await?;
            Ok(())
        }

        async fn replace_index_data(
            &self,
            index: &CodebaseIndex,
            output: &IndexBuildOutput,
        ) -> Result<(), ReferenceCodebaseStorageError> {
            let mut tx = self.pool.begin().await?;

            sqlx::query("DELETE FROM reference_codebase_files WHERE index_id = $1")
                .bind(index.id)
                .execute(&mut *tx)
                .await?;

            for f in &output.files {
                insert_file(&mut tx, index, f).await?;
            }
            for s in &output.symbols {
                insert_symbol(&mut tx, index, s).await?;
            }
            for e in &output.edges {
                insert_edge(&mut tx, index, e).await?;
            }
            for c in &output.chunks {
                insert_chunk(&mut tx, index, c).await?;
            }

            sqlx::query(
                r#"
                UPDATE reference_codebase_indexes
                SET language_set = $2,
                    file_count = $3,
                    symbol_count = $4,
                    chunk_count = $5,
                    edge_count = $6,
                    updated_at = NOW()
                WHERE id = $1
                "#,
            )
            .bind(index.id)
            .bind(&output.language_set)
            .bind(
                output
                    .files
                    .iter()
                    .filter(|f| f.skipped_reason.is_none())
                    .count() as i32,
            )
            .bind(output.symbols.len() as i32)
            .bind(output.chunks.len() as i32)
            .bind(output.edges.len() as i32)
            .execute(&mut *tx)
            .await?;

            tx.commit().await?;
            Ok(())
        }

        async fn insert_embedding(
            &self,
            index: &CodebaseIndex,
            chunk: &CodebaseChunk,
            embedding_model: &str,
            embedding_dim: i32,
            embedding: &[f32],
        ) -> Result<(), ReferenceCodebaseStorageError> {
            let literal = vector_literal(embedding);
            sqlx::query(
                r#"
                INSERT INTO reference_codebase_embeddings (
                    chunk_id, index_id, tenant_id, workspace_id, document_id,
                    document_repo_id, algorithm_id, embedding_model, embedding_dim, embedding
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    embedding = EXCLUDED.embedding
                "#,
            )
            .bind(chunk.id)
            .bind(index.id)
            .bind(index.tenant_id)
            .bind(index.workspace_id)
            .bind(&index.document_id)
            .bind(index.document_repo_id)
            .bind(chunk.algorithm_id)
            .bind(embedding_model)
            .bind(embedding_dim)
            .bind(literal)
            .execute(&self.pool)
            .await?;
            Ok(())
        }

        async fn get_index(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            index_id: Uuid,
        ) -> Result<Option<CodebaseIndex>, ReferenceCodebaseStorageError> {
            let row = sqlx::query(
                "SELECT * FROM reference_codebase_indexes WHERE tenant_id = $1 AND workspace_id = $2 AND id = $3",
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(index_id)
            .fetch_optional(&self.pool)
            .await?;
            row.map(decode_index).transpose()
        }

        async fn list_indexes_for_repo(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            document_repo_id: Uuid,
        ) -> Result<Vec<CodebaseIndex>, ReferenceCodebaseStorageError> {
            let rows = sqlx::query(
                r#"
                SELECT * FROM reference_codebase_indexes
                WHERE tenant_id = $1 AND workspace_id = $2 AND document_repo_id = $3
                ORDER BY created_at DESC
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_repo_id)
            .fetch_all(&self.pool)
            .await?;
            rows.into_iter().map(decode_index).collect()
        }

        async fn anchor_symbols_for_artifact(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            index_id: Uuid,
            code_artifact_id: Uuid,
        ) -> Result<Vec<Uuid>, ReferenceCodebaseStorageError> {
            // Match by file_path + overlapping line range. An artifact
            // can resolve to multiple symbols (e.g. a two-impl struct);
            // the caller decides what to do with the seed set.
            let rows = sqlx::query(
                r#"
                SELECT s.id
                FROM reference_codebase_symbols s
                JOIN code_artifacts ca
                  ON ca.id = $4
                 AND ca.tenant_id = s.tenant_id
                 AND ca.workspace_id = s.workspace_id
                 AND ca.file_path = s.file_path
                 AND ca.start_line <= s.end_line
                 AND ca.end_line >= s.start_line
                WHERE s.tenant_id = $1
                  AND s.workspace_id = $2
                  AND s.index_id = $3
                ORDER BY s.start_line ASC
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(index_id)
            .bind(code_artifact_id)
            .fetch_all(&self.pool)
            .await?;
            rows.iter()
                .map(|r| r.try_get::<Uuid, _>("id").map_err(Into::into))
                .collect()
        }

        async fn symbol_ids_by_name(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            index_id: Uuid,
            symbol_name: &str,
            limit: i64,
        ) -> Result<Vec<Uuid>, ReferenceCodebaseStorageError> {
            let rows = sqlx::query(
                r#"
                SELECT id
                FROM reference_codebase_symbols
                WHERE tenant_id = $1
                  AND workspace_id = $2
                  AND index_id = $3
                  AND (name = $4 OR LOWER(name) = LOWER($4))
                ORDER BY (name = $4) DESC, start_line ASC
                LIMIT $5
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(index_id)
            .bind(symbol_name)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;
            rows.iter()
                .map(|r| r.try_get::<Uuid, _>("id").map_err(Into::into))
                .collect()
        }

        async fn top_symbols_by_degree(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            index_id: Uuid,
            limit: i64,
        ) -> Result<Vec<Uuid>, ReferenceCodebaseStorageError> {
            // Symbol-level degree: incoming + outgoing edges that have a
            // symbol on both sides. Ignores file-only edges.
            let rows = sqlx::query(
                r#"
                WITH edge_endpoints AS (
                    SELECT source_symbol_id AS sid FROM reference_codebase_edges
                    WHERE index_id = $3 AND source_symbol_id IS NOT NULL
                    UNION ALL
                    SELECT target_symbol_id AS sid FROM reference_codebase_edges
                    WHERE index_id = $3 AND target_symbol_id IS NOT NULL
                )
                SELECT s.id, COUNT(e.sid) AS degree
                FROM reference_codebase_symbols s
                LEFT JOIN edge_endpoints e ON e.sid = s.id
                WHERE s.tenant_id = $1
                  AND s.workspace_id = $2
                  AND s.index_id = $3
                GROUP BY s.id
                ORDER BY degree DESC, s.start_line ASC
                LIMIT $4
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(index_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;
            rows.iter()
                .map(|r| r.try_get::<Uuid, _>("id").map_err(Into::into))
                .collect()
        }

        async fn fetch_subgraph(
            &self,
            tenant_id: Uuid,
            workspace_id: Uuid,
            index_id: Uuid,
            seed_symbol_ids: &[Uuid],
            hops: i32,
            max_nodes: i32,
        ) -> Result<CodebaseSubgraph, ReferenceCodebaseStorageError> {
            if seed_symbol_ids.is_empty() {
                return Ok(CodebaseSubgraph {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                    truncated: false,
                });
            }

            // Recursive CTE walks the symbol-edge graph outward from the
            // seed set. Depth caps at `hops`; the outer LIMIT clips at
            // `max_nodes` so a dense node doesn't explode the response.
            // Undirected: a `calls` edge from A→B pulls B in whether we
            // started at A or at B. UNION ALL + per-node MIN(depth) in
            // the wrapper picks the shortest path from the seed set.
            //
            // `max_nodes + 1` is fetched so we can tell whether we
            // truncated; the extra row is not returned.
            let node_limit = max_nodes as i64 + 1;
            let node_rows = sqlx::query(
                r#"
                WITH RECURSIVE seed(symbol_id, depth) AS (
                    SELECT UNNEST($4::uuid[]) AS symbol_id, 0 AS depth
                ),
                walk(symbol_id, depth) AS (
                    SELECT symbol_id, depth FROM seed
                    UNION
                    SELECT CASE
                             WHEN e.source_symbol_id = w.symbol_id THEN e.target_symbol_id
                             ELSE e.source_symbol_id
                           END AS symbol_id,
                           w.depth + 1 AS depth
                    FROM walk w
                    JOIN reference_codebase_edges e
                      ON e.index_id = $3
                     AND (
                            (e.source_symbol_id = w.symbol_id AND e.target_symbol_id IS NOT NULL)
                         OR (e.target_symbol_id = w.symbol_id AND e.source_symbol_id IS NOT NULL)
                     )
                    WHERE w.depth < $5
                ),
                visited AS (
                    SELECT symbol_id, MIN(depth) AS depth
                    FROM walk
                    WHERE symbol_id IS NOT NULL
                    GROUP BY symbol_id
                )
                SELECT
                    s.id              AS symbol_id,
                    s.name            AS name,
                    s.qualified_name  AS qualified_name,
                    s.symbol_kind     AS kind,
                    s.language        AS language,
                    s.file_path       AS file_path,
                    s.start_line      AS start_line,
                    s.end_line        AS end_line,
                    v.depth           AS depth,
                    c.id              AS chunk_id,
                    c.code_artifact_id IS NOT NULL AS is_anchor,
                    COALESCE(c.algorithm_focus, 0.0) AS algorithm_focus
                FROM visited v
                JOIN reference_codebase_symbols s
                  ON s.id = v.symbol_id
                 AND s.tenant_id = $1
                 AND s.workspace_id = $2
                 AND s.index_id = $3
                LEFT JOIN LATERAL (
                    SELECT id, code_artifact_id, algorithm_focus
                    FROM reference_codebase_chunks
                    WHERE symbol_id = s.id
                    ORDER BY (chunk_kind = 'algorithm_anchor') DESC,
                             algorithm_focus DESC
                    LIMIT 1
                ) c ON true
                ORDER BY v.depth ASC, is_anchor DESC, s.start_line ASC
                LIMIT $6
                "#,
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(index_id)
            .bind(seed_symbol_ids)
            .bind(hops)
            .bind(node_limit)
            .fetch_all(&self.pool)
            .await?;

            let truncated = node_rows.len() as i64 > max_nodes as i64;
            let keep = node_rows.len().min(max_nodes as usize);
            let mut nodes = Vec::with_capacity(keep);
            let mut node_ids: Vec<Uuid> = Vec::with_capacity(keep);
            for row in node_rows.iter().take(keep) {
                let symbol_id: Uuid = row.try_get("symbol_id")?;
                node_ids.push(symbol_id);
                nodes.push(SubgraphNode {
                    symbol_id,
                    name: row.try_get("name")?,
                    qualified_name: row.try_get("qualified_name")?,
                    kind: row.try_get("kind")?,
                    language: row.try_get("language")?,
                    file_path: row.try_get("file_path")?,
                    start_line: row.try_get("start_line")?,
                    end_line: row.try_get("end_line")?,
                    depth: row.try_get("depth")?,
                    chunk_id: row.try_get::<Option<Uuid>, _>("chunk_id")?,
                    is_anchor: row.try_get("is_anchor")?,
                    algorithm_focus: row
                        .try_get::<f32, _>("algorithm_focus")
                        .unwrap_or(0.0),
                });
            }

            // Edges strictly between symbols that survived the node
            // truncation. Filter in SQL so we don't ship edges the UI
            // would drop anyway.
            let edge_rows = if node_ids.is_empty() {
                Vec::new()
            } else {
                sqlx::query(
                    r#"
                    SELECT DISTINCT
                        source_symbol_id,
                        target_symbol_id,
                        edge_type AS kind,
                        target_name
                    FROM reference_codebase_edges
                    WHERE index_id = $1
                      AND source_symbol_id = ANY($2::uuid[])
                      AND target_symbol_id = ANY($2::uuid[])
                    "#,
                )
                .bind(index_id)
                .bind(&node_ids)
                .fetch_all(&self.pool)
                .await?
            };

            let mut edges = Vec::with_capacity(edge_rows.len());
            for row in edge_rows {
                edges.push(SubgraphEdge {
                    source_symbol_id: row.try_get("source_symbol_id")?,
                    target_symbol_id: row.try_get("target_symbol_id")?,
                    kind: row.try_get("kind")?,
                    target_name: row.try_get("target_name")?,
                });
            }

            Ok(CodebaseSubgraph {
                nodes,
                edges,
                truncated,
            })
        }
    }

    async fn insert_file(
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        index: &CodebaseIndex,
        f: &CodebaseFile,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO reference_codebase_files (
                id, index_id, tenant_id, workspace_id, document_repo_id,
                file_path, language, checksum, line_count, byte_count, skipped_reason
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            "#,
        )
        .bind(f.id)
        .bind(index.id)
        .bind(index.tenant_id)
        .bind(index.workspace_id)
        .bind(index.document_repo_id)
        .bind(&f.file_path)
        .bind(&f.language)
        .bind(&f.checksum)
        .bind(f.line_count)
        .bind(f.byte_count)
        .bind(f.skipped_reason.as_deref())
        .execute(&mut **tx)
        .await?;
        Ok(())
    }

    async fn insert_symbol(
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        index: &CodebaseIndex,
        s: &CodebaseSymbol,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO reference_codebase_symbols (
                id, index_id, file_id, tenant_id, workspace_id, document_repo_id,
                symbol_kind, name, qualified_name, parent_symbol_id, file_path,
                language, start_line, end_line, start_byte, end_byte
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
            "#,
        )
        .bind(s.id)
        .bind(index.id)
        .bind(s.file_id)
        .bind(index.tenant_id)
        .bind(index.workspace_id)
        .bind(index.document_repo_id)
        .bind(&s.symbol_kind)
        .bind(&s.name)
        .bind(&s.qualified_name)
        .bind(s.parent_symbol_id)
        .bind(&s.file_path)
        .bind(&s.language)
        .bind(s.start_line)
        .bind(s.end_line)
        .bind(s.start_byte)
        .bind(s.end_byte)
        .execute(&mut **tx)
        .await?;
        Ok(())
    }

    async fn insert_edge(
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        index: &CodebaseIndex,
        e: &CodebaseEdge,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO reference_codebase_edges (
                index_id, tenant_id, workspace_id, document_repo_id, edge_type,
                source_symbol_id, target_symbol_id, source_file_id, target_file_id,
                target_name, metadata
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            "#,
        )
        .bind(index.id)
        .bind(index.tenant_id)
        .bind(index.workspace_id)
        .bind(index.document_repo_id)
        .bind(&e.edge_type)
        .bind(e.source_symbol_id)
        .bind(e.target_symbol_id)
        .bind(e.source_file_id)
        .bind(e.target_file_id)
        .bind(e.target_name.as_deref())
        .bind(&e.metadata)
        .execute(&mut **tx)
        .await?;
        Ok(())
    }

    async fn insert_chunk(
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        index: &CodebaseIndex,
        c: &CodebaseChunk,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO reference_codebase_chunks (
                id, index_id, file_id, symbol_id, algorithm_id, code_artifact_id,
                tenant_id, workspace_id, document_id, document_repo_id,
                chunk_kind, language, file_path, symbol_name, start_line, end_line,
                token_estimate, algorithm_focus, content, content_hash
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
            "#,
        )
        .bind(c.id)
        .bind(index.id)
        .bind(c.file_id)
        .bind(c.symbol_id)
        .bind(c.algorithm_id)
        .bind(c.code_artifact_id)
        .bind(index.tenant_id)
        .bind(index.workspace_id)
        .bind(&index.document_id)
        .bind(index.document_repo_id)
        .bind(&c.chunk_kind)
        .bind(&c.language)
        .bind(&c.file_path)
        .bind(c.symbol_name.as_deref())
        .bind(c.start_line)
        .bind(c.end_line)
        .bind(c.token_estimate)
        .bind(c.algorithm_focus)
        .bind(&c.content)
        .bind(&c.content_hash)
        .execute(&mut **tx)
        .await?;
        Ok(())
    }

    fn decode_index(row: PgRow) -> Result<CodebaseIndex, ReferenceCodebaseStorageError> {
        let mode_s: String = row.try_get("mode")?;
        let status_s: String = row.try_get("status")?;
        Ok(CodebaseIndex {
            id: row.try_get("id")?,
            tenant_id: row.try_get("tenant_id")?,
            workspace_id: row.try_get("workspace_id")?,
            document_id: row.try_get("document_id")?,
            document_repo_id: row.try_get("document_repo_id")?,
            repo_url: row.try_get("repo_url")?,
            repo_commit: row.try_get("repo_commit")?,
            repo_path: row.try_get("repo_path")?,
            repo_license: row.try_get("repo_license")?,
            mode: CodebaseIndexMode::parse(&mode_s).ok_or_else(|| {
                ReferenceCodebaseStorageError::Decode(format!("bad mode: {mode_s}"))
            })?,
            status: CodebaseIndexStatus::parse(&status_s).ok_or_else(|| {
                ReferenceCodebaseStorageError::Decode(format!("bad status: {status_s}"))
            })?,
            language_set: row.try_get("language_set")?,
            file_count: row.try_get("file_count")?,
            symbol_count: row.try_get("symbol_count")?,
            chunk_count: row.try_get("chunk_count")?,
            edge_count: row.try_get("edge_count")?,
            error_message: row.try_get("error_message")?,
            started_at: row.try_get::<Option<DateTime<Utc>>, _>("started_at")?,
            completed_at: row.try_get::<Option<DateTime<Utc>>, _>("completed_at")?,
        })
    }

    fn vector_literal(v: &[f32]) -> String {
        let mut s = String::with_capacity(v.len() * 10 + 2);
        s.push('[');
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            use std::fmt::Write;
            let _ = write!(s, "{:.7}", x);
        }
        s.push(']');
        s
    }
}
