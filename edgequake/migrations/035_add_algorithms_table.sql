-- Migration 035: Add algorithms table for algorithm extraction extension.
--
-- Stores structured algorithm definitions extracted from documents via 3-pass LLM pipeline.
-- Scoped by tenant_id + workspace_id for multi-tenant isolation.

SET search_path = public;

CREATE TABLE IF NOT EXISTS algorithms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    workspace_id UUID NOT NULL,
    document_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    steps JSONB NOT NULL DEFAULT '[]',
    inputs JSONB NOT NULL DEFAULT '[]',
    outputs JSONB NOT NULL DEFAULT '[]',
    preconditions JSONB NOT NULL DEFAULT '[]',
    complexity TEXT,
    mathematical_notation TEXT,
    pseudocode TEXT,
    tags JSONB NOT NULL DEFAULT '[]',
    confidence TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'pending',
    verification_status TEXT,
    verification_details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT algorithms_valid_status CHECK (status IN ('pending', 'approved', 'rejected')),
    CONSTRAINT algorithms_valid_confidence CHECK (confidence IN ('high', 'medium', 'low'))
);

-- Multi-tenant isolation index
CREATE INDEX IF NOT EXISTS idx_algorithms_tenant_workspace
    ON algorithms(tenant_id, workspace_id);

-- Fast lookups by source document
CREATE INDEX IF NOT EXISTS idx_algorithms_document
    ON algorithms(document_id);

-- Status filtering (e.g., list pending for review)
CREATE INDEX IF NOT EXISTS idx_algorithms_status
    ON algorithms(tenant_id, workspace_id, status);

-- Full-text search on algorithm name
CREATE INDEX IF NOT EXISTS idx_algorithms_name_fts
    ON algorithms USING gin (to_tsvector('english', name));

-- Update task_type constraint to include 'algorithm_extraction'
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS valid_task_type;
ALTER TABLE tasks ADD CONSTRAINT valid_task_type CHECK (
    task_type IN ('upload', 'insert', 'scan', 'reindex', 'pdf_processing', 'algorithm_extraction')
);
