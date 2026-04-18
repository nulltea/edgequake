-- Migration 042: Reference Code GraphRAG — document-repo detection storage.
--
-- Stores candidate reference-implementation repositories detected for each
-- ingested document. Populated by the `repo_detection` task, which runs:
--   Layer A — PDF hyperlink + reference-section filtering (edgequake-pdf).
--   Layer B — SearXNG + Crawl4AI + LLM fallback (edgequake-agents).
--
-- Scoped by tenant_id + workspace_id for multi-tenant isolation.

SET search_path = public;

-- One row per detected candidate repo. Re-detection is idempotent via the
-- unique constraint on (tenant, workspace, document, host, owner, repo).
CREATE TABLE IF NOT EXISTS document_repos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    workspace_id UUID NOT NULL,
    document_id TEXT NOT NULL,

    -- Canonical repo identity.
    host TEXT NOT NULL,              -- github | gitlab | bitbucket
    owner TEXT NOT NULL,
    repo TEXT NOT NULL,
    url TEXT NOT NULL,

    -- Provenance.
    detection_method TEXT NOT NULL,  -- pdf_link | web_search
    pdf_page_index INT,              -- Layer A only
    search_rank INT,                 -- Layer B only (0-based)
    source_url TEXT,                 -- Layer B only (SearXNG result URL)

    -- Scoring + review.
    confidence TEXT NOT NULL DEFAULT 'medium',  -- high | medium | low
    status TEXT NOT NULL DEFAULT 'pending',     -- pending | approved | rejected

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT document_repos_valid_host CHECK (host IN ('github', 'gitlab', 'bitbucket')),
    CONSTRAINT document_repos_valid_method CHECK (detection_method IN ('pdf_link', 'web_search')),
    CONSTRAINT document_repos_valid_confidence CHECK (confidence IN ('high', 'medium', 'low')),
    CONSTRAINT document_repos_valid_status CHECK (status IN ('pending', 'approved', 'rejected')),
    CONSTRAINT document_repos_unique_per_doc UNIQUE (tenant_id, workspace_id, document_id, host, owner, repo)
);

CREATE INDEX IF NOT EXISTS idx_document_repos_tenant_workspace
    ON document_repos(tenant_id, workspace_id);

CREATE INDEX IF NOT EXISTS idx_document_repos_document
    ON document_repos(document_id);

CREATE INDEX IF NOT EXISTS idx_document_repos_status
    ON document_repos(tenant_id, workspace_id, status);


-- One row per (doc) tracking detection-job state. Lets us distinguish
-- "detection ran and found nothing" from "detection not yet attempted",
-- and record the most-recent error for UX.
CREATE TABLE IF NOT EXISTS document_repo_detections (
    tenant_id UUID NOT NULL,
    workspace_id UUID NOT NULL,
    document_id TEXT NOT NULL,

    status TEXT NOT NULL,            -- running | complete | failed
    layer_a_candidates INT NOT NULL DEFAULT 0,
    layer_b_candidates INT NOT NULL DEFAULT 0,
    error_message TEXT,

    attempted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    PRIMARY KEY (tenant_id, workspace_id, document_id),
    CONSTRAINT document_repo_detections_valid_status CHECK (status IN ('running', 'complete', 'failed'))
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
        'repo_detection'
    )
);
