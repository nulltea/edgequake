-- Extend the valid_extraction_method check constraint on pdf_documents
-- to include 'kreuzberg' (Kreuzberg library: PDFium + RT-DETR v2 + TATR) and
-- 'oarocr' (OAR-OCR: PP-DocLayout_plus-L + PP-OCRv5).
--
-- Also keep 'pdfextract' valid for backward-compat with any existing rows
-- written before migration 038. Storage layer parses 'pdfextract' as Kreuzberg.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = 'pdf_documents'
    ) THEN
        IF EXISTS (
            SELECT 1
            FROM information_schema.table_constraints
            WHERE table_schema = 'public'
              AND table_name = 'pdf_documents'
              AND constraint_name = 'valid_extraction_method'
        ) THEN
            ALTER TABLE pdf_documents
                DROP CONSTRAINT valid_extraction_method;
        END IF;

        ALTER TABLE pdf_documents
            ADD CONSTRAINT valid_extraction_method CHECK (
                extraction_method IS NULL OR
                extraction_method IN ('text', 'vision', 'hybrid', 'edgeparse', 'pdfextract', 'kreuzberg', 'oarocr')
            );
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'pdf_records_extraction_method_check'
    ) THEN
        ALTER TABLE pdf_records
            DROP CONSTRAINT pdf_records_extraction_method_check;

        ALTER TABLE pdf_records
            ADD CONSTRAINT pdf_records_extraction_method_check
            CHECK (extraction_method IN ('text', 'vision', 'hybrid', 'edgeparse', 'pdfextract', 'kreuzberg', 'oarocr'));
    END IF;
END $$;
