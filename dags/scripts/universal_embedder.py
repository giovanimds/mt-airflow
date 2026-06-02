import os
import json
import logging
import time
import traceback
import re
import psycopg2
from psycopg2.extras import execute_values
from openai import OpenAI, RateLimitError, AuthenticationError
import httpx
import redis
import requests

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

# Mistral embed model supports up to 8192 tokens
# We use 7850 to be safe (leaving 342 tokens margin for tokenizer differences)
# The actual limit reported by API is 7900, but we leave room for:
# - Tokenizer BOS/EOS tokens that may be added during decode/encode cycles
# - Tokenization differences between our tokenizer and Mistral's API tokenizer
MAX_TOKENS_PER_CHUNK = 7850
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "25"))  # Reduced from 32 to avoid rate limits

# Global tokenizer cache
_TOKENIZER = None


def get_db_connection(dbname):
    """Get database connection."""
    params = {
        "host": os.environ.get("PG_HOST", "postgres.default.svc.cluster.local"),
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


def get_tokenizer():
    """Get or create Mistral tokenizer for accurate token counting."""
    global _TOKENIZER
    
    if _TOKENIZER is not None:
        return _TOKENIZER
    
    # Try to use sentencepiece if available (used by Mistral tokenizer)
    try:
        import sentencepiece as spm
        
        # Download mistral tokenizer model file
        tokenizer_url = "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model"
        tokenizer_path = "/tmp/mistral_tokenizer.model"
        
        response = requests.get(tokenizer_url, timeout=30)
        response.raise_for_status()
        
        with open(tokenizer_path, "wb") as f:
            f.write(response.content)
        
        _TOKENIZER = spm.SentencePieceProcessor()
        _TOKENIZER.load(tokenizer_path)
        
        log.info("✅ Tokenizer Mistral (SentencePiece) baixado e carregado com sucesso")
        return _TOKENIZER
        
    except ImportError:
        log.warning("SentencePiece não disponível, tentando HuggingFace tokenizers...")
    except Exception as e:
        log.error(f"⚠️  Falha ao baixar tokenizer SentencePiece: {e}")
        log.warning("Tentando HuggingFace tokenizers...")
    
    # Try HuggingFace tokenizers library
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import WhitespaceSplit
        
        # Download mistral tokenizer.json and vocab.json
        tokenizer_url = "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json"
        tokenizer_path = "/tmp/mistral_tokenizer.json"
        vocab_url = "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/vocab.json"
        vocab_path = "/tmp/mistral_vocab.json"
        
        response = requests.get(tokenizer_url, timeout=30)
        response.raise_for_status()
        with open(tokenizer_path, "wb") as f:
            f.write(response.content)
        
        response = requests.get(vocab_url, timeout=30)
        response.raise_for_status()
        with open(vocab_path, "wb") as f:
            f.write(response.content)
        
        _TOKENIZER = Tokenizer(BPE(vocab_path, tokenizer_path))
        _TOKENIZER.pre_tokenizer = WhitespaceSplit()
        
        log.info("✅ Tokenizer Mistral (HuggingFace) baixado e carregado com sucesso")
        return _TOKENIZER
        
    except ImportError:
        log.warning("HuggingFace tokenizers não disponível")
    except Exception as e:
        log.error(f"⚠️  Falha ao baixar tokenizer HuggingFace: {e}")
    
    # Ultimate fallback: use a simple token approximation
    log.warning("Usando fallback de contagem de tokens baseada em regex (menos precisa)")
    
    class SimpleTokenizer:
        def __init__(self):
            # Approximation based on common tokenization patterns
            self.pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|\w+|[^\\w\\s]|")
        
        def encode(self, text):
            tokens = self.pattern.findall(text.lower())
            return type('Encoding', (), {'ids': list(range(len(tokens))), 'tokens': tokens})()
        
        def decode(self, ids):
            # For fallback, just return the original text
            return ""
    
    _TOKENIZER = SimpleTokenizer()
    return _TOKENIZER


def count_tokens(text, token_ids=None):
    """Count tokens in text using Mistral tokenizer.
    
    If token_ids is provided, use its length directly (most accurate).
    Otherwise, encode the text.
    """
    if not text or not isinstance(text, str):
        return 0
    
    # If we already have token IDs, use them directly
    if token_ids is not None:
        return len(token_ids)
    
    tokenizer = get_tokenizer()
    encoding = tokenizer.encode(text)
    
    # Handle different return types
    if hasattr(encoding, 'ids'):
        return len(encoding.ids)
    elif isinstance(encoding, list):
        return len(encoding)
    else:
        # Fallback
        return len(text.split())


def split_text_into_chunks(text, max_tokens=MAX_TOKENS_PER_CHUNK):
    """
    Split text into chunks by actual token count.
    Returns list of chunks, each with <= max_tokens tokens.
    Uses Mistral tokenizer for accurate token counting.
    """
    if not text or not isinstance(text, str):
        return []
    
    tokenizer = get_tokenizer()
    
    try:
        # Try to encode the full text
        encoding = tokenizer.encode(text)
        
        # Get token IDs
        if hasattr(encoding, 'ids'):
            token_ids = encoding.ids
        elif isinstance(encoding, list):
            token_ids = encoding
        else:
            # Fallback to word-based splitting
            words = text.split()
            chunks = []
            for i in range(0, len(words), max_tokens):
                chunks.append(" ".join(words[i:i+max_tokens]))
            return chunks
        
        # Split into chunks based on token count
        chunks = []
        for i in range(0, len(token_ids), max_tokens):
            chunk_token_ids = token_ids[i:i + max_tokens]
            
            # Decode the chunk back to text if possible
            # IMPORTANT: Use decode methods that don't add special tokens
            if hasattr(tokenizer, 'decode_ids'):
                # SentencePieceProcessor: use decode_ids to avoid adding BOS/EOS
                chunk_text = tokenizer.decode_ids(chunk_token_ids)
            elif hasattr(tokenizer, 'decode'):
                # HuggingFace Tokenizer: skip special tokens
                chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
            else:
                # Fallback: join token IDs as strings (not ideal but works)
                chunk_text = " ".join(str(tid) for tid in chunk_token_ids)
            
            # Store chunk with its token count (we already have the exact count from the slice)
            chunks.append((chunk_text, len(chunk_token_ids)))
        
        # Extract just the text from tuples and validate
        result_chunks = []
        for chunk_text, token_count in chunks:
            # This should never exceed max_tokens since we sliced by max_tokens
            # But double-check to be safe
            if token_count > max_tokens:
                log.warning(f"⚠️  Chunk tem {token_count} tokens (max: {max_tokens}), truncando...")
                # Take only the first max_tokens from this chunk
                # Re-encode to get proper text
                re_encoded = tokenizer.encode(chunk_text)
                if hasattr(re_encoded, 'ids'):
                    truncated_ids = re_encoded.ids[:max_tokens]
                elif isinstance(re_encoded, list):
                    truncated_ids = re_encoded[:max_tokens]
                else:
                    truncated_ids = re_encoded
                
                if hasattr(tokenizer, 'decode_ids'):
                    chunk_text = tokenizer.decode_ids(truncated_ids)
                elif hasattr(tokenizer, 'decode'):
                    chunk_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
                else:
                    chunk_text = " ".join(str(tid) for tid in truncated_ids)
            
            result_chunks.append(chunk_text)
        
        return result_chunks
        
    except Exception as e:
        log.error(f"⚠️  Erro ao dividir texto em chunks: {e}")
        # Ultimate fallback: split by words
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_tokens):
            chunks.append(" ".join(words[i:i+max_tokens]))
        return chunks


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


def generate_embeddings(client, model, texts, batch_size=EMBEDDING_BATCH_SIZE):
    """
    Generate embeddings for a list of texts using Mistral API.
    Handles batching, retries, and token-limit auto-splitting automatically.
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Retry logic with exponential backoff
        for attempt in range(5):  # Max 5 attempts
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                batch_embeddings = [e.embedding for e in response.data]
                all_embeddings.extend(batch_embeddings)
                time.sleep(1)  # Rate limiting: 1 second delay between batches to avoid API rate limits
                break  # Success, exit retry loop
                
            except RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                log.warning(f"⚠️  Rate limit hit (attempt {attempt + 1}/5). Waiting {wait_time}s...")
                time.sleep(wait_time)
                
            except AuthenticationError as e:
                log.error(f"❌ Authentication error: {e}")
                raise
                
            except Exception as e:
                # Check if it is a "Too many tokens overall" error
                err_msg = str(e).lower()
                is_too_many_tokens = (
                    "too many tokens" in err_msg or 
                    "split into more batches" in err_msg or
                    ("400" in err_msg and "token" in err_msg)
                )
                if hasattr(e, 'status_code') and getattr(e, 'status_code') == 400:
                    is_too_many_tokens = True
                
                if is_too_many_tokens and len(batch) > 1:
                    log.warning(f"⚠️  Batch contains too many tokens overall. Splitting batch of size {len(batch)} into two halves...")
                    mid = len(batch) // 2
                    left_batch = batch[:mid]
                    right_batch = batch[mid:]
                    try:
                        # Recursively generate embeddings for both halves with half batch size
                        left_embeddings = generate_embeddings(client, model, left_batch, batch_size=len(left_batch))
                        right_embeddings = generate_embeddings(client, model, right_batch, batch_size=len(right_batch))
                        all_embeddings.extend(left_embeddings)
                        all_embeddings.extend(right_embeddings)
                        break  # Success, exit retry loop
                    except Exception as sub_e:
                        log.error(f"❌ Error when processing split batches: {sub_e}")
                        raise
                
                log.error(f"❌ Unexpected error (attempt {attempt + 1}/5): {e}")
                if attempt == 4:  # Last attempt
                    raise
                time.sleep(2 ** attempt)
    
    return all_embeddings


def run_db_transaction_with_retry(db_name, transaction_func, max_retries=10, delay=3):
    for attempt in range(max_retries):
        conn = None
        try:
            conn = get_db_connection(db_name)
            with conn:
                with conn.cursor() as cur:
                    result = transaction_func(cur)
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return result
        except (psycopg2.Error, psycopg2.InterfaceError, psycopg2.OperationalError) as e:
            log.warning(f"Database error on attempt {attempt + 1}/{max_retries} for {db_name}: {e}")
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Failed to execute database transaction after {max_retries} attempts.")


def process_batch(batch):
    """Process a batch of items from the queue."""
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

    # Get Mistral client
    client, model = get_mistral_client()

    for db_name, tables in grouped.items():
        try:
            # 1. Fetch texts that need embedding
            texts_to_embed_by_table = {}
            
            def fetch_txn(cur):
                ensure_embeddings_chunks_table(cur)
                
                table_data = {}
                for table_name, items in tables.items():
                    source_col = items[0].get("source_column", "text")
                    ids = [item.get("id") for item in items]
                    id_placeholders = ",".join(["%s"] * len(ids))
                    
                    if table_name == 'corpus':
                        cur.execute(
                            f"SELECT id, title, text, metadata FROM public.corpus WHERE id IN ({id_placeholders});",
                            tuple(ids)
                        )
                    else:
                        cur.execute(
                            f"SELECT id, {source_col} FROM {table_name} WHERE id IN ({id_placeholders});",
                            tuple(ids)
                        )
                    
                    rows = cur.fetchall()
                    if rows:
                        table_data[table_name] = rows
                return table_data

            texts_to_embed_by_table = run_db_transaction_with_retry(db_name, fetch_txn)
            
            if not texts_to_embed_by_table:
                continue

            # 2. Generate embeddings (no DB connection is held open during external API calls!)
            all_chunks_to_save = []
            corpus_embeddings_to_update = []
            
            for table_name, rows in texts_to_embed_by_table.items():
                for row_data in rows:
                    if table_name == 'corpus':
                        row_id, title, text, metadata = row_data
                    else:
                        row_id, text = row_data
                        title, metadata = None, None
                        
                    if not text or not isinstance(text, str) or len(text.strip()) == 0:
                        log.warning(f"⚠️  Texto vazio para {table_name}.{row_id}, pulando")
                        continue
                    
                    # Split text into chunks
                    chunks = split_text_into_chunks(text)
                    
                    if not chunks:
                        log.warning(f"⚠️  Nenhum chunk gerado para {table_name}.{row_id}")
                        continue
                    
                    # Log chunk sizes for debugging
                    # Use count_tokens with text to get accurate count (may include BOS in some tokenizers)
                    # But this should never exceed MAX_TOKENS_PER_CHUNK due to our splitting logic
                    for idx, chunk in enumerate(chunks):
                        token_count = count_tokens(chunk)
                        if token_count > MAX_TOKENS_PER_CHUNK:
                            # This should NOT happen with our fix, but log if it does
                            log.error(f"❌ CRITICAL: Chunk {idx} para {table_name}.{row_id} tem {token_count} tokens (max: {MAX_TOKENS_PER_CHUNK})")
                            log.error(f"   Chunk text (first 200 chars): {chunk[:200]}...")
                    
                    try:
                        # Generate embeddings for all chunks
                        chunk_embeddings = generate_embeddings(client, model, chunks)
                        
                        if not chunk_embeddings:
                            log.error(f"❌ Nenhum embedding gerado para {table_name}.{row_id}")
                            continue
                        
                        if len(chunk_embeddings) != len(chunks):
                            log.error(f"❌ Mismatch: {len(chunk_embeddings)} embeddings vs {len(chunks)} chunks para {table_name}.{row_id}")
                            continue
                        
                        # Save ALL chunks to embeddings_chunks table
                        for idx, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                            all_chunks_to_save.append((row_id, table_name, idx, chunk_text, embedding))
                        
                        log.info(f"✅ {len(chunks)} chunks processados para {table_name}.{row_id}")
                        
                        # Handle metadata + domain embedding for corpus table
                        if table_name == 'corpus':
                            meta_dict = {}
                            if metadata:
                                if isinstance(metadata, str):
                                    try:
                                        meta_dict = json.loads(metadata)
                                    except Exception:
                                        pass
                                elif isinstance(metadata, dict):
                                    meta_dict = metadata
                            
                            domain = meta_dict.get("domain", "")
                            topic = meta_dict.get("topic", "")
                            difficulty = meta_dict.get("difficulty", "")
                            
                            parts = []
                            if title:
                                parts.append(f"Title: {title}")
                            if domain:
                                parts.append(f"Domain: {domain}")
                            if topic:
                                parts.append(f"Topic: {topic}")
                            if difficulty:
                                parts.append(f"Difficulty: {difficulty}")
                                
                            metadata_text = " | ".join(parts)
                            if metadata_text:
                                meta_embeddings = generate_embeddings(client, model, [metadata_text])
                                if meta_embeddings:
                                    corpus_embeddings_to_update.append((meta_embeddings[0], row_id))
                                    log.info(f"✅ Embedding de metadados gerado para corpus.{row_id}")
                        
                    except Exception as e:
                        log.error(f"❌ Erro ao gerar embeddings para {table_name}.{row_id}: {e}")
                        traceback.print_exc()
                        continue
            
            # 3. Batch save all chunks at once (using a new connection with retries)
            if all_chunks_to_save or corpus_embeddings_to_update:
                def save_txn(cur):
                    if all_chunks_to_save:
                        # Delete existing chunks for these record_ids
                        record_ids = list(set([data[0] for data in all_chunks_to_save]))
                        cur.execute(
                            "DELETE FROM public.embeddings_chunks WHERE record_id = ANY(%s::uuid[]);",
                            (record_ids,)
                        )
                        execute_values(
                            cur,
                            """INSERT INTO public.embeddings_chunks 
                                (record_id, table_name, chunk_index, chunk_text, embedding) 
                                VALUES %s""",
                            all_chunks_to_save,
                            template=None,
                            page_size=100
                        )
                        log.info(f"✅ {len(all_chunks_to_save)} chunks salvos na tabela embeddings_chunks")
                    
                    if corpus_embeddings_to_update:
                        execute_values(
                            cur,
                            """UPDATE public.corpus AS c
                               SET embedding = val.embedding::float8[]::vector
                               FROM (VALUES %s) AS val(embedding, id)
                               WHERE c.id = val.id::uuid""",
                            corpus_embeddings_to_update,
                            page_size=100
                        )
                        log.info(f"✅ {len(corpus_embeddings_to_update)} embeddings de metadados atualizados na tabela corpus")

                run_db_transaction_with_retry(db_name, save_txn)
                        
        except Exception as e:
            log.error(f"❌ Erro ao processar batch para o DB {db_name}: {e}")
            traceback.print_exc()


def get_mistral_client():
    """Get or create Mistral API client (OpenAI-compatible)."""
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    base_url = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
    model = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(30.0, connect=10.0),
        max_retries=3,
    )
    return client, model


def main():
    log.info(f"Iniciando Universal Embedder. Conectando ao Redis em {REDIS_URL}...")
    r = redis.Redis.from_url(REDIS_URL, socket_timeout=POLL_TIMEOUT_SEC + 5)
    
    # Testar conexão
    r.ping()
    log.info("Conexão com Redis/Valkey estabelecida com sucesso.")
    
    # Test Mistral API connection on startup
    try:
        client, model = get_mistral_client()
        response = client.embeddings.create(model=model, input=["test"])
        log.info(f"✅ Conexão com Mistral API ({model}) verificada com sucesso.")
    except Exception as e:
        log.error(f"❌ Falha ao conectar à Mistral API: {e}")
    
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
