-- Rename 'oarocrvl' → 'vlmocr' in extraction_method check constraints and existing data.

DO $$
BEGIN
    -- Update existing rows in pdf_documents
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = 'pdf_documents'
    ) THEN
        UPDATE pdf_documents SET extraction_method = 'vlmocr'
        WHERE extraction_method = 'oarocrvl';

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
                extraction_method IN ('text', 'vision', 'hybrid', 'edgeparse', 'kreuzberg', 'oarocr', 'vlmocr')
            );
    END IF;

    -- Update existing rows in pdf_records
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = 'pdf_records'
    ) THEN
        UPDATE pdf_records SET extraction_method = 'vlmocr'
        WHERE extraction_method = 'oarocrvl';

        IF EXISTS (
            SELECT 1
            FROM information_schema.table_constraints
            WHERE constraint_name = 'pdf_records_extraction_method_check'
        ) THEN
            ALTER TABLE pdf_records
                DROP CONSTRAINT pdf_records_extraction_method_check;

            ALTER TABLE pdf_records
                ADD CONSTRAINT pdf_records_extraction_method_check
                CHECK (extraction_method IN ('text', 'vision', 'hybrid', 'edgeparse', 'kreuzberg', 'oarocr', 'vlmocr'));
        END IF;
    END IF;
END $$;
