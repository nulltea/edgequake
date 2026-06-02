-- Migration 056: Per-document free-form label.
--
-- Adds `label` to `documents`. Users assign a short, human-meaningful tag
-- (e.g. the paper's method codename — "Nexus", "AloePri", "DP-SGD") so the
-- documents listing can display it next to the title in brackets, the MCP
-- tools can surface it, and downstream UIs can group / filter by it.
--
-- Single label per document. Free text, length-bounded to keep the title
-- column readable (1..=80 chars). NULL means "no label assigned".

ALTER TABLE documents ADD COLUMN IF NOT EXISTS label TEXT NULL;

-- Length guardrail. Empty strings are disallowed: callers should send NULL
-- (or DELETE the row) to clear the label.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'documents_label_length'
    ) THEN
        ALTER TABLE documents
            ADD CONSTRAINT documents_label_length
            CHECK (label IS NULL OR char_length(label) BETWEEN 1 AND 80);
    END IF;
END$$;

-- Index for workspace-scoped label filtering / grouping. Partial — most
-- documents will have NULL labels and we don't want to bloat the index.
CREATE INDEX IF NOT EXISTS idx_documents_workspace_label
    ON documents(tenant_id, workspace_id, label)
    WHERE label IS NOT NULL;
