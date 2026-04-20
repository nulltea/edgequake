-- Migration 049: allow `manual` as a detection_method on `document_repos`.
--
-- Before this change, `detection_method` CHECK restricted the column to
-- ('pdf_link', 'web_search'). The References tab now lets users add a
-- reference repo by pasting a URL (bypassing Layer A link-parsing and
-- Layer B web-search), so rows need a third enum value. Rows created by
-- the new /repos/add endpoint get `manual`.
--
-- Idempotent: drops + recreates the constraint; safe to re-run.

SET search_path = public;

ALTER TABLE document_repos
    DROP CONSTRAINT IF EXISTS document_repos_valid_method;

ALTER TABLE document_repos
    ADD CONSTRAINT document_repos_valid_method
    CHECK (detection_method IN ('pdf_link', 'web_search', 'manual'));
