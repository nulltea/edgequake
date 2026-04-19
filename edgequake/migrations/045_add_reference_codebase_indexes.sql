-- Migration 045: Reference codebase RAG indexes.
--
-- Phase 2 of Reference Code GraphRAG. These tables intentionally model a
-- separate codebase RAG instance for approved reference repositories rather
-- than mixing full-repo chunks with paper chunks or Phase 1 code_artifacts.

SET search_path = public;

CREATE TABLE IF NOT EXISTS reference_codebase_indexes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID NOT NULL,
    workspace_id        UUID NOT NULL,
    document_id         TEXT NOT NULL,
    document_repo_id    UUID NOT NULL REFERENCES document_repos(id) ON DELETE CASCADE,

    repo_url            TEXT NOT NULL,
    repo_commit         TEXT NOT NULL,
    repo_path           TEXT NOT NULL,
    repo_license        TEXT,

    mode                TEXT NOT NULL DEFAULT 'algorithm_focused',
    status              TEXT NOT NULL DEFAULT 'queued',
    language_set        TEXT[] NOT NULL DEFAULT '{}',

    file_count          INT NOT NULL DEFAULT 0,
    symbol_count        INT NOT NULL DEFAULT 0,
    chunk_count         INT NOT NULL DEFAULT 0,
    edge_count          INT NOT NULL DEFAULT 0,

    error_message       TEXT,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT reference_codebase_indexes_valid_mode CHECK (mode IN ('algorithm_focused', 'full')),
    CONSTRAINT reference_codebase_indexes_valid_status CHECK (
        status IN ('queued','scanning','parsing','chunking','embedding','complete','failed')
    ),
    CONSTRAINT reference_codebase_indexes_unique_commit UNIQUE (
        tenant_id, workspace_id, document_repo_id, repo_commit, mode
    )
);

CREATE INDEX IF NOT EXISTS idx_reference_codebase_indexes_repo
    ON reference_codebase_indexes(tenant_id, workspace_id, document_repo_id);

CREATE INDEX IF NOT EXISTS idx_reference_codebase_indexes_status
    ON reference_codebase_indexes(status);


CREATE TABLE IF NOT EXISTS reference_codebase_files (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    index_id            UUID NOT NULL REFERENCES reference_codebase_indexes(id) ON DELETE CASCADE,
    tenant_id           UUID NOT NULL,
    workspace_id        UUID NOT NULL,
    document_repo_id    UUID NOT NULL,

    file_path           TEXT NOT NULL,
    language            TEXT NOT NULL,
    checksum            TEXT NOT NULL,
    line_count          INT NOT NULL DEFAULT 0,
    byte_count          INT NOT NULL DEFAULT 0,
    skipped_reason      TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT reference_codebase_files_unique_path UNIQUE(index_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_reference_codebase_files_repo
    ON reference_codebase_files(tenant_id, workspace_id, document_repo_id);


CREATE TABLE IF NOT EXISTS reference_codebase_symbols (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    index_id            UUID NOT NULL REFERENCES reference_codebase_indexes(id) ON DELETE CASCADE,
    file_id             UUID NOT NULL REFERENCES reference_codebase_files(id) ON DELETE CASCADE,
    tenant_id           UUID NOT NULL,
    workspace_id        UUID NOT NULL,
    document_repo_id    UUID NOT NULL,

    symbol_kind         TEXT NOT NULL,
    name                TEXT NOT NULL,
    qualified_name      TEXT NOT NULL,
    parent_symbol_id    UUID,
    file_path           TEXT NOT NULL,
    language            TEXT NOT NULL,
    start_line          INT NOT NULL,
    end_line            INT NOT NULL,
    start_byte          INT NOT NULL DEFAULT 0,
    end_byte            INT NOT NULL DEFAULT 0,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT reference_codebase_symbols_line_order CHECK (end_line >= start_line)
);

CREATE INDEX IF NOT EXISTS idx_reference_codebase_symbols_index
    ON reference_codebase_symbols(index_id);
CREATE INDEX IF NOT EXISTS idx_reference_codebase_symbols_name
    ON reference_codebase_symbols(tenant_id, workspace_id, document_repo_id, name);


CREATE TABLE IF NOT EXISTS reference_codebase_edges (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    index_id            UUID NOT NULL REFERENCES reference_codebase_indexes(id) ON DELETE CASCADE,
    tenant_id           UUID NOT NULL,
    workspace_id        UUID NOT NULL,
    document_repo_id    UUID NOT NULL,

    edge_type           TEXT NOT NULL,
    source_symbol_id    UUID REFERENCES reference_codebase_symbols(id) ON DELETE CASCADE,
    target_symbol_id    UUID REFERENCES reference_codebase_symbols(id) ON DELETE CASCADE,
    source_file_id      UUID REFERENCES reference_codebase_files(id) ON DELETE CASCADE,
    target_file_id      UUID REFERENCES reference_codebase_files(id) ON DELETE CASCADE,
    target_name         TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT reference_codebase_edges_has_source CHECK (
        source_symbol_id IS NOT NULL OR source_file_id IS NOT NULL
    )
);

CREATE INDEX IF NOT EXISTS idx_reference_codebase_edges_source_symbol
    ON reference_codebase_edges(source_symbol_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_reference_codebase_edges_target_symbol
    ON reference_codebase_edges(target_symbol_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_reference_codebase_edges_index
    ON reference_codebase_edges(index_id, edge_type);


CREATE TABLE IF NOT EXISTS reference_codebase_chunks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    index_id            UUID NOT NULL REFERENCES reference_codebase_indexes(id) ON DELETE CASCADE,
    file_id             UUID NOT NULL REFERENCES reference_codebase_files(id) ON DELETE CASCADE,
    symbol_id           UUID REFERENCES reference_codebase_symbols(id) ON DELETE SET NULL,
    algorithm_id        UUID REFERENCES algorithms(id) ON DELETE SET NULL,
    code_artifact_id    UUID REFERENCES code_artifacts(id) ON DELETE SET NULL,

    tenant_id           UUID NOT NULL,
    workspace_id        UUID NOT NULL,
    document_id         TEXT NOT NULL,
    document_repo_id    UUID NOT NULL,

    chunk_kind          TEXT NOT NULL,
    language            TEXT NOT NULL,
    file_path           TEXT NOT NULL,
    symbol_name         TEXT,
    start_line          INT NOT NULL,
    end_line            INT NOT NULL,
    token_estimate      INT NOT NULL DEFAULT 0,
    algorithm_focus     REAL NOT NULL DEFAULT 0,
    content             TEXT NOT NULL,
    content_hash        TEXT NOT NULL,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT reference_codebase_chunks_line_order CHECK (end_line >= start_line),
    CONSTRAINT reference_codebase_chunks_unique_hash UNIQUE(index_id, file_path, start_line, end_line, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_reference_codebase_chunks_index
    ON reference_codebase_chunks(index_id);
CREATE INDEX IF NOT EXISTS idx_reference_codebase_chunks_algorithm
    ON reference_codebase_chunks(algorithm_id);
CREATE INDEX IF NOT EXISTS idx_reference_codebase_chunks_repo
    ON reference_codebase_chunks(tenant_id, workspace_id, document_repo_id);


ALTER TABLE tasks DROP CONSTRAINT IF EXISTS valid_task_type;
ALTER TABLE tasks ADD CONSTRAINT valid_task_type CHECK (
    task_type IN (
        'upload',
        'insert',
        'scan',
        'reindex',
        'pdf_processing',
        'algorithm_extraction',
        'algorithm_embedding',
        'repo_detection',
        'code_reference_analysis',
        'reference_codebase_index'
    )
);
