import os
import json
import logging
import time
import traceback
import psycopg2
from psycopg2.extras import execute_values
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://valkey-primary.default.svc.cluster.local:6379")
QUEUE_NAME = os.environ.get("EMBEDDER_QUEUE_NAME", "embedder_queue")
BATCH_SIZE = int(os.environ.get("EMBEDDER_BATCH_SIZE", "100"))
POLL_TIMEOUT_SEC = int(os.environ.get("EMBEDDER_POLL_TIMEOUT", "5"))
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "1"))

# Configure text splitter for chunking long texts
# Mistral embed model supports up to 8192 tokens, so we use 8000 to be safe
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=8000,
    chunk_overlap=200,
    length_function=lambda text: len(text.split()),
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
)


def get_db_connection(dbname):
    params = {
        "host": os.environ.get("PG_HOST", "postgres.morescotech.com.br"),
        "port": int(os.environ.get("PG_PORT", 5432)),
        "user": os.environ.get("PG_USER", "yugabyte"),
        "password": os.environ.get("PG_PASSWORD", "YugabytePass2026"),
        "database": dbname,
        "sslmode": "disable"
    }
    try:
        return psycopg2.connect(**params, load_balance=True)
    except (TypeError, psycopg2.Error):
        return psycopg2.connect(**params)


def split_text_into_chunks(text, max_tokens=8000):
    """
    Split text into chunks using RecursiveCharacterTextSplitter.
    Returns list of chunks, each with <= max_tokens tokens.
    """
    if not text or not isinstance(text, str):
        return []
    
    docs = TEXT_SPLITTER.create_documents([text])
    chunks = [doc.page_content for doc in docs]
    
    # Further split if any chunk is still too large
    final_chunks = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) <= max_tokens:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(words), max_tokens):
                final_chunks.append(" ".join(words[i:i+max_tokens]))
    
    return final_chunks


def ensure_embeddings_chunks_table(cur):
    """Ensure the embeddings_chunks partitioned table exists."""
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'embeddings_chunks'
        );
    """)
    if not cur.fetchone()[0]:
        cur.execute("""
            CREATE TABLE public.embeddings_chunks (
                id UUID DEFAULT gen_random_uuid(),
                record_id UUID NOT NULL,
                table_name TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding FLOAT[] NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            ) PARTITION BY LIST (table_name);
            
            CREATE TABLE IF NOT EXISTS public.embeddings_chunks_corpus 
                PARTITION OF public.embeddings_chunks 
                FOR VALUES IN ('corpus');
                
            CREATE TABLE IF NOT EXISTS public.embeddings_chunks_qa_dataset 
                PARTITION OF public.embeddings_chunks 
                FOR VALUES IN ('qa_dataset');
                
            CREATE TABLE IF NOT EXISTS public.embeddings_chunks_other 
                PARTITION OF public.embeddings_chunks 
                DEFAULT;
                
            CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_corpus_record ON public.embeddings_chunks_corpus(record_id);
            CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_corpus_chunk ON public.embeddings_chunks_corpus(record_id, chunk_index);
            
            CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_qa_record ON public.embeddings_chunks_qa_dataset(record_id);
            CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_qa_chunk ON public.embeddings_chunks_qa_dataset(record_id, chunk_index);
            
            CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_other_record ON public.embeddings_chunks_other(record_id);
            CREATE INDEX IF NOT EXISTS idx_embeddings_chunks_other_chunk ON public.embeddings_chunks_other(record_id, chunk_index);
        """)
        log.info("✅ Tabela embeddings_chunks e suas partições criadas")


def process_batch(batch):
    if not batch:
        return

    log.info(f"Processando batch de {len(batch)} itens.")
    
    # Group by database and table to optimize connections and queries
    grouped = {}
    for item in batch:
        db = item.get("database", "ai_labs")
        table = item.get("table", "corpus")
        if db not in grouped:
            grouped[db] = {}
        if table not in grouped[db]:
            grouped[db][table] = []
        grouped[db][table].append(item)

    embedder = MistralAIEmbeddings(
        mistral_api_key=os.environ.get("MISTRAL_API_KEY", ""),
        model="mistral-embed",
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS
    )

    for db_name, tables in grouped.items():
        try:
            # 1. Fetch texts that need embedding (using a short-lived DB connection)
            texts_to_embed_by_table = {}
            with get_db_connection(db_name) as conn:
                with conn.cursor() as cur:
                    ensure_embeddings_chunks_table(cur)
                    conn.commit()
                    
                    for table_name, items in tables.items():
                        source_col = items[0].get("source_column", "text")
                        
                        ids = [item.get("id") for item in items]
                        id_placeholders = ",".join(["%s"] * len(ids))
                        
                        # Fetch texts that need embedding
                        cur.execute(
                            f"SELECT id, {source_col} FROM {table_name} WHERE id IN ({id_placeholders});",
                            tuple(ids)
                        )
                        rows = cur.fetchall()
                        if rows:
                            texts_to_embed_by_table[table_name] = rows
            
            if not texts_to_embed_by_table:
                continue

            # 2. Generate embeddings (no DB connection is held open during external API calls!)
            all_chunks_to_save = []
            
            for table_name, rows in texts_to_embed_by_table.items():
                for row_id, text in rows:
                    if not text or not isinstance(text, str) or len(text.strip()) == 0:
                        log.warning(f"⚠️  Texto vazio para {table_name}.{row_id}, pulando")
                        continue
                    
                    # Split text into chunks
                    chunks = split_text_into_chunks(text)
                    
                    if not chunks:
                        log.warning(f"⚠️  Nenhum chunk gerado para {table_name}.{row_id}")
                        continue
                    
                    try:
                        # Generate embeddings for all chunks
                        chunk_embeddings = embedder.embed_documents(chunks)
                        
                        log.debug(f"DEBUG: chunk_embeddings type: {type(chunk_embeddings)}, len: {len(chunk_embeddings) if chunk_embeddings else 0}")
                        
                        if not chunk_embeddings:
                            log.error(f"❌ Nenhum embedding gerado para {table_name}.{row_id}")
                            continue
                        
                        # Save ALL chunks to embeddings_chunks table
                        for idx, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                            log.debug(f"DEBUG: chunk {idx}, embedding type: {type(embedding)}, len: {len(embedding) if embedding else 0}")
                            all_chunks_to_save.append((row_id, table_name, idx, chunk_text, embedding))
                        
                        log.info(f"✅ {len(chunks)} chunks processados para {table_name}.{row_id}")
                        
                    except Exception as e:
                        log.error(f"❌ Erro ao gerar embeddings para {table_name}.{row_id}: {e}")
                        traceback.print_exc()
                        continue
            
            # 3. Batch save all chunks at once (using a new short-lived DB connection)
            if all_chunks_to_save:
                with get_db_connection(db_name) as conn:
                    with conn.cursor() as cur:
                        log.info(f"DEBUG: Total chunks to save: {len(all_chunks_to_save)}")
                        
                        # Delete existing chunks for these record_ids
                        record_ids = list(set([data[0] for data in all_chunks_to_save]))
                        cur.execute(
                            "DELETE FROM public.embeddings_chunks WHERE record_id = ANY(%s::uuid[]);",
                            (record_ids,)
                        )
                        log.info(f"DEBUG: Deleted existing chunks for {len(record_ids)} records")
                        
                        # Use execute_values instead of execute_batch
                        execute_values(
                            cur,
                            """INSERT INTO public.embeddings_chunks 
                                (record_id, table_name, chunk_index, chunk_text, embedding) 
                                VALUES %s""",
                            all_chunks_to_save,
                            template=None,
                            page_size=100
                        )
                        conn.commit()
                        log.info(f"✅ {len(all_chunks_to_save)} chunks salvos na tabela embeddings_chunks")
                        
        except Exception as e:
            log.error(f"❌ Erro ao processar batch para o DB {db_name}: {e}")
            traceback.print_exc()


def main():
    log.info(f"Iniciando Universal Embedder. Conectando ao Redis em {REDIS_URL}...")
    r = redis.Redis.from_url(REDIS_URL, socket_timeout=POLL_TIMEOUT_SEC + 5)
    
    # Testar conexão
    r.ping()
    log.info("Conexão com Redis/Valkey estabelecida com sucesso.")
    
    while True:
        try:
            batch = []
            # Get first item with blocking (BRPOP)
            result = r.brpop(QUEUE_NAME, timeout=POLL_TIMEOUT_SEC)
            
            if result:
                _, item_json = result
                batch.append(json.loads(item_json))
                
                # Get rest of batch without blocking (RPOP)
                while len(batch) < BATCH_SIZE:
                    item_json = r.rpop(QUEUE_NAME)
                    if item_json:
                        batch.append(json.loads(item_json))
                    else:
                        break
                
                process_batch(batch)
                
        except redis.exceptions.TimeoutError:
            continue
        except redis.exceptions.ConnectionError:
            log.warning("Conexão com Valkey perdida. Tentando reconectar...")
            time.sleep(2)
        except Exception as e:
            log.error(f"❌ Erro no loop principal: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
