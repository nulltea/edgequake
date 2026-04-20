-- Migration 050: LLM verification columns on document_repos.
--
-- After Layer A or Layer B produces a candidate, a verifier LLM inspects
-- the repository's README against the paper's title/authors/abstract and
-- assigns a verdict. These columns carry that verdict; the main
-- `confidence` column and `status` review state remain untouched by this
-- migration (the app-side orchestrator will mutate `confidence` to 'low'
-- when verdict='unrelated' — see the plan doc for rationale).
--
-- All four columns are NULL-default so pre-existing rows aren't
-- considered verified until a re-detect fills them in.

SET search_path = public;

ALTER TABLE document_repos
    ADD COLUMN IF NOT EXISTS verification_verdict TEXT
        CHECK (verification_verdict IS NULL
               OR verification_verdict IN
                  ('official', 'third_party', 'unrelated', 'inconclusive')),
    ADD COLUMN IF NOT EXISTS verification_confidence REAL
        CHECK (verification_confidence IS NULL
               OR (verification_confidence >= 0
                   AND verification_confidence <= 1)),
    ADD COLUMN IF NOT EXISTS verification_rationale TEXT,
    ADD COLUMN IF NOT EXISTS verified_at TIMESTAMPTZ;

-- Helpful for the UI's "flag unrelated candidates" query.
CREATE INDEX IF NOT EXISTS idx_document_repos_verification_verdict
    ON document_repos (verification_verdict)
    WHERE verification_verdict IS NOT NULL;
