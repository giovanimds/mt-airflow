-- Create corpus_embeddings table for chunked embeddings
-- This table stores individual chunks of long texts that exceed the 8192 token limit

CREATE TABLE IF NOT EXISTS public.corpus_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_corpus 
        FOREIGN KEY (corpus_id) 
        REFERENCES public.corpus(id) 
        ON DELETE CASCADE
);

-- Create index on corpus_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_corpus_embeddings_corpus_id ON public.corpus_embeddings(corpus_id);

-- Create index on chunk_index for ordering
CREATE INDEX IF NOT EXISTS idx_corpus_embeddings_chunk_index ON public.corpus_embeddings(chunk_index);

-- Create composite index for corpus_id + chunk_index
CREATE INDEX IF NOT EXISTS idx_corpus_embeddings_corpus_chunk ON public.corpus_embeddings(corpus_id, chunk_index);

-- Create index on created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_corpus_embeddings_created_at ON public.corpus_embeddings(created_at);

-- Verify table creation
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name = 'corpus_embeddings';

-- Verify indexes
SELECT indexname, indexdef FROM pg_indexes 
WHERE tablename = 'corpus_embeddings' AND schemaname = 'public';
