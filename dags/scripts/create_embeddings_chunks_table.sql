-- Create partitioned table for all embeddings chunks
-- Partitioned by table_name for scalability

-- Create partitioned table by table_name only (simpler and more effective)
DROP TABLE IF EXISTS public.embeddings_chunks CASCADE;
CREATE TABLE IF NOT EXISTS public.embeddings_chunks (
    id UUID DEFAULT gen_random_uuid(),
    record_id UUID NOT NULL,
    table_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
) PARTITION BY LIST (table_name);

-- Create partitions for known tables
CREATE TABLE IF NOT EXISTS public.embeddings_chunks_corpus 
    PARTITION OF public.embeddings_chunks 
    FOR VALUES IN ('corpus');

CREATE TABLE IF NOT EXISTS public.embeddings_chunks_qa_dataset 
    PARTITION OF public.embeddings_chunks 
    FOR VALUES IN ('qa_dataset');

-- Create a default partition for any other table names
CREATE TABLE IF NOT EXISTS public.embeddings_chunks_other 
    PARTITION OF public.embeddings_chunks 
    DEFAULT;

-- Create indexes on each partition for performance
-- Index on record_id for each partition
CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_corpus_record ON public.embeddings_chunks_corpus(record_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_corpus_chunk ON public.embeddings_chunks_corpus(record_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_qa_record ON public.embeddings_chunks_qa_dataset(record_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_qa_chunk ON public.embeddings_chunks_qa_dataset(record_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_other_record ON public.embeddings_chunks_other(record_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_other_chunk ON public.embeddings_chunks_other(record_id, chunk_index);

-- Table and partitions created successfully
