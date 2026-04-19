-- Migration 043: Reference Code GraphRAG Phase 1 — code_artifacts + runs.
--
-- Stores functions / classes that the code-analyzer sidecar identified as
-- implementing a paper's extracted Algorithm, plus per-analysis job state.
-- Populated after a user approves a document_repos row and the resulting
-- CodeReferenceAnalysis task runs to completion.

SET search_path = public;

-- One row per (algorithm, function) match. Candidates start `pending`;
-- after human review an Approved row gets materialised into the AGE graph
-- and its embedding is retained; Rejected rows keep the artifact metadata
-- but their embedding is removed from the vector store.
CREATE TABLE IF NOT EXISTS code_artifacts (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID NOT NULL,
    workspace_id        UUID NOT NULL,
    document_id         TEXT NOT NULL,
    algorithm_id        UUID NOT NULL REFERENCES algorithms(id) ON DELETE CASCADE,
    document_repo_id    UUID NOT NULL REFERENCES document_repos(id) ON DELETE CASCADE,

    -- Identity of the cloned repo snapshot.
    repo_commit         TEXT NOT NULL,   -- resolved SHA at clone time
    repo_license        TEXT,            -- SPDX identifier if detectable

    -- Location.
    language            TEXT NOT NULL,   -- tree-sitter grammar name
    file_path           TEXT NOT NULL,   -- relative to repo root
    symbol_name         TEXT,            -- function/class/method name, null if unknown
    start_line          INT NOT NULL,    -- 1-indexed inclusive
    end_line            INT NOT NULL,
    snippet             TEXT NOT NULL,   -- code text for display (bounded)

    -- Scoring from the analyzer.
    match_rationale     TEXT,
    match_confidence    TEXT NOT NULL DEFAULT 'medium',

    -- Review state.
    status              TEXT NOT NULL DEFAULT 'pending',
    embedding_id        UUID,            -- pointer into workspace vector store; null = not embedded yet

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT code_artifacts_valid_status CHECK (status IN ('pending', 'approved', 'rejected')),
    CONSTRAINT code_artifacts_valid_confidence CHECK (match_confidence IN ('high', 'medium', 'low')),
    CONSTRAINT code_artifacts_line_order CHECK (end_line >= start_line),

    -- Natural key: one row per (algo, repo, file+range). Lets re-analysis
    -- upsert without losing the user's review decision.
    CONSTRAINT code_artifacts_unique_per_match UNIQUE (
        tenant_id, workspace_id, algorithm_id, document_repo_id,
        file_path, start_line, end_line
    )
);

CREATE INDEX IF NOT EXISTS idx_code_artifacts_algorithm_status
    ON code_artifacts(algorithm_id, status);
CREATE INDEX IF NOT EXISTS idx_code_artifacts_document
    ON code_artifacts(tenant_id, workspace_id, document_id);
CREATE INDEX IF NOT EXISTS idx_code_artifacts_repo
    ON code_artifacts(document_repo_id);


-- Per-(doc, repo) analysis-run tracking. Mirrors document_repo_detections
-- from migration 042.
CREATE TABLE IF NOT EXISTS code_reference_runs (
    tenant_id           UUID NOT NULL,
    workspace_id        UUID NOT NULL,
    document_id         TEXT NOT NULL,
    document_repo_id    UUID NOT NULL REFERENCES document_repos(id) ON DELETE CASCADE,

    status              TEXT NOT NULL,       -- queued|cloning|analyzing|embedding|awaiting_review|complete|failed
    algorithm_count     INT NOT NULL DEFAULT 0,
    finding_count       INT NOT NULL DEFAULT 0,
    cost_usd_equivalent REAL,                -- claude CLI's total_cost_usd (subscription = theoretical)
    error_message       TEXT,

    attempted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,

    PRIMARY KEY (tenant_id, workspace_id, document_id, document_repo_id),
    CONSTRAINT code_reference_runs_valid_status CHECK (
        status IN ('queued','cloning','analyzing','embedding','awaiting_review','complete','failed')
    )
);


-- Register the new task type.
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
        'code_reference_analysis'
    )
);
