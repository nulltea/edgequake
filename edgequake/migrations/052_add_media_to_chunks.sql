-- Migration 052: Add media columns to chunks for PDF figure storage.
--
-- VLM-OCR extracts figures (Image/Chart/Seal layout elements) from PDFs. Each
-- figure becomes its own chunk: caption + image bytes. media_bytes holds the
-- PNG; media_mime tags the encoding; kind distinguishes text vs figure rows so
-- retrieval can label hits as figures; figure_id is the stable extractor-side
-- identifier (`fig_{page}_{order_index}`) that ties the chunk back to the
-- markdown placeholder it replaced.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS media_bytes BYTEA;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS media_mime  TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS kind        TEXT NOT NULL DEFAULT 'text';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS figure_id   TEXT;

-- Partial index speeds up figure-only lookups (e.g. the media-fetch endpoint
-- queries by (document_id, figure_id)). Text rows skipped to keep the index small.
CREATE INDEX IF NOT EXISTS idx_chunks_figure
    ON chunks(document_id, figure_id)
    WHERE kind = 'figure';
