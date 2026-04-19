-- Migration 046: Reference codebase chunk embeddings.
--
-- Separate vector table for full reference-codebase RAG. Kept separate from
-- code_artifact_embeddings because Phase 1 approved snippets and Phase 2
-- coding-agent context have different lifecycle and retrieval semantics.

SET search_path = public;

CREATE TABLE IF NOT EXISTS reference_codebase_embeddings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id            UUID NOT NULL UNIQUE
                         REFERENCES reference_codebase_chunks(id) ON DELETE CASCADE,
    index_id            UUID NOT NULL REFERENCES reference_codebase_indexes(id) ON DELETE CASCADE,
    tenant_id           UUID NOT NULL,
    workspace_id        UUID NOT NULL,
    document_id         TEXT NOT NULL,
    document_repo_id    UUID NOT NULL,
    algorithm_id        UUID,
    embedding_model     TEXT NOT NULL,
    embedding_dim       INT NOT NULL,
    embedding           vector(896) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reference_codebase_embeddings_scope
    ON reference_codebase_embeddings(tenant_id, workspace_id, document_repo_id, index_id);

CREATE INDEX IF NOT EXISTS idx_reference_codebase_embeddings_algorithm
    ON reference_codebase_embeddings(algorithm_id);

CREATE INDEX IF NOT EXISTS idx_reference_codebase_embeddings_hnsw
    ON reference_codebase_embeddings
    USING hnsw (embedding vector_cosine_ops);
