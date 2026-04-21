-- Phase 2 follow-up: rich symbol metadata for agent-facing retrieval.
--
-- Tree-sitter extracts parameters, return types, docstrings, visibility, and
-- async/test flags per symbol. Storing these as JSONB avoids per-language
-- schema churn — each extractor emits whatever the grammar exposes.
--
-- Surface in query_code hits and graph-node payloads so an LLM agent can
-- see a signature + docstring without re-reading the function body.

ALTER TABLE reference_codebase_symbols
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

-- GIN index on metadata for future attribute-based filtering (is_async,
-- visibility='public', etc.). Low write cost at index build time, no runtime
-- impact until a query actually filters on metadata fields.
CREATE INDEX IF NOT EXISTS idx_reference_codebase_symbols_metadata
    ON reference_codebase_symbols USING GIN (metadata);
