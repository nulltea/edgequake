-- Migration 041: Add algorithm_type column.
--
-- Distinguishes the kind of algorithmic construct each row represents
-- (Algorithm, Protocol, Functionality, Theorem, …) so the UI can render a
-- type-specific badge and users can filter/search by type.
--
-- Values are intentionally open text (not an enum): crypto / ML / systems
-- papers use different vocabularies and the LLM extracts the natural-language
-- label from the paper itself. The frontend renders a small set of common
-- values as color-coded badges and falls back to a neutral badge otherwise.

SET search_path = public;

ALTER TABLE algorithms
    ADD COLUMN IF NOT EXISTS algorithm_type TEXT NOT NULL DEFAULT 'Algorithm';

-- Keep the default so existing rows get a non-null value; new inserts can
-- (and should) specify their own type.
