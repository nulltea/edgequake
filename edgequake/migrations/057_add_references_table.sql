-- Migration 057: Add references table for citation/reference parsing.
--
-- Stores numbered references parsed (regex/heuristic, no LLM) from a
-- document's reference section. Scoped by tenant_id + workspace_id for
-- multi-tenant isolation. Deterministic parser output — no review/approval
-- status; rows are overwritten when a document is reprocessed.

SET search_path = public;

CREATE TABLE IF NOT EXISTS document_references (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    workspace_id UUID NOT NULL,
    document_id TEXT NOT NULL,
    reference_number INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    doi TEXT,
    url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Multi-tenant isolation index.
CREATE INDEX IF NOT EXISTS idx_document_references_tenant_workspace
    ON document_references(tenant_id, workspace_id);

-- Fast lookups by source document (list endpoint + retrieval enrichment),
-- ordered by reference number.
CREATE INDEX IF NOT EXISTS idx_document_references_document
    ON document_references(document_id, reference_number);
