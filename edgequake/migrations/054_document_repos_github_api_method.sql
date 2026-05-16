-- Migration 054: allow `github_api` as a detection_method on `document_repos`.
--
-- Layer B's web-search arm split into two parallel paths:
-- - `github_api`: GitHub Search API via octocrab (primary).
-- - `web_search`: SearXNG + Crawl4AI fallback for niche/non-GitHub hits.
--
-- Rows produced by the new GitHub-API path are tagged `github_api` so the
-- UI can distinguish them. Existing rows keep their original tag.
--
-- Idempotent: drops + recreates the constraint; safe to re-run.

SET search_path = public;

ALTER TABLE document_repos
    DROP CONSTRAINT IF EXISTS document_repos_valid_method;

ALTER TABLE document_repos
    ADD CONSTRAINT document_repos_valid_method
    CHECK (detection_method IN ('pdf_link', 'web_search', 'github_api', 'manual'));
