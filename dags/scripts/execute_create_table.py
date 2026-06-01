#!/usr/bin/env python3
import psycopg2
import sys

def execute_sql():
    try:
        conn = psycopg2.connect(
            host="postgres.morescotech.com.br",
            port=5432,
            user="yugabyte",
            password="YugabytePass2026",
            database="ai_labs",
            sslmode="disable"
        )
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Create table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.corpus_embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    corpus_id UUID NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding FLOAT[] NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("✅ Tabela corpus_embeddings criada")
            
            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_corpus_embeddings_corpus_id 
                ON public.corpus_embeddings(corpus_id);
            """)
            print("✅ Índice idx_corpus_embeddings_corpus_id criado")
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_corpus_embeddings_chunk_index 
                ON public.corpus_embeddings(chunk_index);
            """)
            print("✅ Índice idx_corpus_embeddings_chunk_index criado")
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_corpus_embeddings_corpus_chunk 
                ON public.corpus_embeddings(corpus_id, chunk_index);
            """)
            print("✅ Índice idx_corpus_embeddings_corpus_chunk criado")
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_corpus_embeddings_created_at 
                ON public.corpus_embeddings(created_at);
            """)
            print("✅ Índice idx_corpus_embeddings_created_at criado")
            
            # Add foreign key
            try:
                cur.execute("""
                    ALTER TABLE public.corpus_embeddings 
                    ADD CONSTRAINT fk_corpus 
                    FOREIGN KEY (corpus_id) 
                    REFERENCES public.corpus(id) 
                    ON DELETE CASCADE;
                """)
                print("✅ Foreign key fk_corpus adicionada")
            except Exception as e:
                print(f"⚠️  Foreign key pode já existir: {e}")
            
            # Verify
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'corpus_embeddings';
            """)
            if cur.fetchone():
                print("\n✅ Tabela corpus_embeddings existe no banco")
            
            cur.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'corpus_embeddings' AND schemaname = 'public';
            """)
            indexes = cur.fetchall()
            print(f"✅ Índices criados: {[idx[0] for idx in indexes]}")
        
        conn.close()
        print("\n✅ Todos os objetos criados com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    execute_sql()
