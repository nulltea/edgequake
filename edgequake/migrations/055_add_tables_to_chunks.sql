-- Migration 055: Add table columns to chunks for VLM-OCR table extraction.
--
-- Companion to migration 052 (figure media). Tables extracted from PDFs land
-- as their own chunk rows with kind='table', addressed by a stable extractor
-- id (`tbl_{page}_{order_index}`). table_html holds the VLM-rendered HTML;
-- table_rows holds the algorithmically-parsed { headers, rows } structure for
-- downstream UIs and classification. table_type / table_classification_rationale
-- are populated by a later classification stage (perf | quality | complexity |
-- other) so they're NULL on insert.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_id                       TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_html                     TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_rows                     JSONB;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_type                     TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_classification_rationale TEXT;

-- Partial index for table-only lookups: list-tables and get-table endpoints
-- both query by (document_id [, table_id]). Text/figure rows are skipped.
CREATE INDEX IF NOT EXISTS idx_chunks_table
    ON chunks(document_id, table_id)
    WHERE kind = 'table';
