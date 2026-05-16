-- Migration 053: Soft-archive support for documents.
--
-- Adds `archived_at` to `documents`. When set, the document keeps its row,
-- PDF, Markdown, algorithms, and document_repos entries, but the archive
-- handler removes chunks, embeddings, KG entity/relationship attribution,
-- and any indexed-code (`reference_codebase_*`) rows derived from it.
--
-- The status CHECK constraint is intentionally untouched: a doc archived
-- after reaching 'indexed' keeps that status so the UI can show it was
-- successfully processed before being archived.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ NULL;

-- Fast path for the Archive page (workspace + archived filter).
CREATE INDEX IF NOT EXISTS idx_documents_archived
    ON documents(tenant_id, workspace_id, archived_at)
    WHERE archived_at IS NOT NULL;

-- Fast path for the main Documents page (workspace + active filter).
CREATE INDEX IF NOT EXISTS idx_documents_active
    ON documents(tenant_id, workspace_id, created_at DESC)
    WHERE archived_at IS NULL;
