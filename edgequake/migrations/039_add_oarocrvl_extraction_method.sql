-- Add 'oarocrvl' to the valid_extraction_method check constraint on pdf_documents.

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
                extraction_method IN ('text', 'vision', 'hybrid', 'edgeparse', 'pdfextract', 'kreuzberg', 'oarocr', 'oarocrvl')
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
            CHECK (extraction_method IN ('text', 'vision', 'hybrid', 'edgeparse', 'pdfextract', 'kreuzberg', 'oarocr', 'oarocrvl'));
    END IF;
END $$;
