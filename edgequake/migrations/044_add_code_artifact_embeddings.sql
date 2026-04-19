-- Migration 044: Phase 1 of the Reference Code GraphRAG extension — vector
-- store for approved code_artifacts.
--
-- A dedicated pgvector table keeps the code-embedder's dimension (896 for
-- jina-code-embeddings-0.5b, 1536 for the 1.5b variant) decoupled from the
-- workspace's text-embedder dimension. One row per (code_artifact_id) —
-- embeddings are written on approve and removed on reject via
-- ON DELETE CASCADE of the parent row.

SET search_path = public;

CREATE TABLE IF NOT EXISTS code_artifact_embeddings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code_artifact_id    UUID NOT NULL UNIQUE
                         REFERENCES code_artifacts(id) ON DELETE CASCADE,
    tenant_id           UUID NOT NULL,
    workspace_id        UUID NOT NULL,
    document_id         TEXT NOT NULL,
    algorithm_id        UUID NOT NULL,
    embedding_model     TEXT NOT NULL,            -- e.g. "jina-code-embeddings"
    embedding_dim       INT NOT NULL,             -- 896 for 0.5b, 1536 for 1.5b
    embedding           vector(896) NOT NULL,     -- Jina 0.5B full width
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_code_artifact_embeddings_tenant_workspace
    ON code_artifact_embeddings(tenant_id, workspace_id);

CREATE INDEX IF NOT EXISTS idx_code_artifact_embeddings_algorithm
    ON code_artifact_embeddings(algorithm_id);

CREATE INDEX IF NOT EXISTS idx_code_artifact_embeddings_document
    ON code_artifact_embeddings(document_id);

-- HNSW on cosine distance. The Jina server returns L2-normalised vectors
-- (confirmed: ‖v‖₂ ≈ 1.0 on a smoke test) so cosine works directly; we
-- keep the cosine operator class for explicitness.
CREATE INDEX IF NOT EXISTS idx_code_artifact_embeddings_hnsw
    ON code_artifact_embeddings
    USING hnsw (embedding vector_cosine_ops);
