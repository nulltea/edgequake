-- Add 'algorithm_embedding' to valid task types
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS valid_task_type;
ALTER TABLE tasks ADD CONSTRAINT valid_task_type CHECK (
    task_type IN ('upload', 'insert', 'scan', 'reindex', 'pdf_processing', 'algorithm_extraction', 'algorithm_embedding')
);
