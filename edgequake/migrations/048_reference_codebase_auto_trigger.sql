-- Migration 048: auto-trigger metadata + parse-error telemetry for Phase 2
-- Reference Codebase RAG.
--
-- Two changes, both idempotent via IF NOT EXISTS.
--
-- 1. reference_codebase_indexes gains:
--    - auto_triggered        (BOOL) — true when the row was inserted by the
--                              code_artifact approval handler, not by an
--                              explicit POST /reference-codebase/indexes
--                              call. Used to apply a lower MAX_FILES cap
--                              and to surface the trigger path in the UI.
--    - max_files_override    (INT)  — per-row cap that takes precedence over
--                              EDGEQUAKE_REFERENCE_CODEBASE_MAX_FILES when
--                              set. Populated (~5000) on auto-triggered
--                              rows so accidentally pointing the indexer at
--                              a 20k-file repo doesn't burn embedding
--                              budget. Explicit POST requests can raise it
--                              via force_reindex + a larger value.
--
-- 2. reference_codebase_files gains:
--    - parse_errors          (INT)  — count of grammar/tree-sitter parse
--                              failures recorded while indexing this file.
--                              Surfaces in the UI as a banner when more
--                              than 5% of files failed — silent parse
--                              crashes look like missing edges, which is
--                              the worst class of "graph is wrong"
--                              debugging.

SET search_path = public;

ALTER TABLE reference_codebase_indexes
    ADD COLUMN IF NOT EXISTS auto_triggered BOOL NOT NULL DEFAULT FALSE;

ALTER TABLE reference_codebase_indexes
    ADD COLUMN IF NOT EXISTS max_files_override INT;

ALTER TABLE reference_codebase_files
    ADD COLUMN IF NOT EXISTS parse_errors INT NOT NULL DEFAULT 0;
