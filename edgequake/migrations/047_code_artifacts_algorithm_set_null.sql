-- Migration 045: preserve code_artifacts across algorithm re-extraction.
--
-- Before this change, `code_artifacts.algorithm_id` was `NOT NULL REFERENCES
-- algorithms(id) ON DELETE CASCADE`. Reprocessing a PDF re-runs
-- `algorithm_extraction`, which calls `delete_algorithms_by_document` before
-- inserting fresh rows; the CASCADE destroyed every related code-reference
-- candidate the user had spent time reviewing.
--
-- We now keep orphaned candidates in the table with `algorithm_id = NULL`
-- so the user can re-link them (or reject them) after reprocessing. The UI
-- can surface these as "orphan" code matches.
--
-- Idempotent: the ALTER/DROP/ADD statements guard on constraint/column
-- existence so re-running the migration is a no-op once applied.

SET search_path = public;

-- 1. Drop the NOT NULL constraint on algorithm_id. After re-extraction the
-- column may be NULL until a user re-links the candidate to a new algorithm.
ALTER TABLE code_artifacts
    ALTER COLUMN algorithm_id DROP NOT NULL;

-- 2. Replace the ON DELETE CASCADE with ON DELETE SET NULL. sqlx/Postgres
-- name the implicit FK `code_artifacts_algorithm_id_fkey`; we drop and
-- recreate it under the same name.
ALTER TABLE code_artifacts
    DROP CONSTRAINT IF EXISTS code_artifacts_algorithm_id_fkey;

ALTER TABLE code_artifacts
    ADD CONSTRAINT code_artifacts_algorithm_id_fkey
    FOREIGN KEY (algorithm_id)
    REFERENCES algorithms(id)
    ON DELETE SET NULL;

-- 3. Relax the UNIQUE constraint so NULL algorithm_id rows don't collide.
-- Postgres treats NULLs as distinct in UNIQUE by default (each NULL is
-- unique), which is what we want: multiple orphaned artifacts from different
-- algorithms can coexist even if their (file_path, start_line, end_line)
-- happen to match.
-- No change needed — the existing
--   UNIQUE (tenant_id, workspace_id, algorithm_id, document_repo_id,
--           file_path, start_line, end_line)
-- already tolerates NULL algorithm_id under default NULLS DISTINCT semantics.

-- 4. Index to find orphans quickly.
CREATE INDEX IF NOT EXISTS idx_code_artifacts_orphan
    ON code_artifacts(tenant_id, workspace_id, document_id)
    WHERE algorithm_id IS NULL;
