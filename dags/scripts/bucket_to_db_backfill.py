import os
import json
import logging
import uuid
import polars as pl
from google.cloud import storage
import psycopg2
from psycopg2.extras import execute_values
import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://valkey-primary.default.svc.cluster.local:6379")

def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("PG_HOST", "postgres.morescotech.com.br"),
        port=int(os.environ.get("PG_PORT", 5432)),
        user=os.environ.get("PG_USER", "yugabyte"),
        password=os.environ.get("PG_PASSWORD", "YugabytePass2026"),
        database=os.environ.get("PG_DATABASE", "ai_labs"),
        sslmode="disable"
    )

def clean_val(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v.replace('\x00', '').replace('\u0000', '')
    return str(v).replace('\x00', '').replace('\u0000', '')

def run_backfill(bucket_name="mt-airflow", prefix="raw_corpus/", out_prefix="datasets/pt-br_Q&A/", limit_files=None, push_to_valkey=True):
    log.info(f"Iniciando smart backfill do bucket gs://{bucket_name}...")
    
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)
    
    # 1. Listar arquivos parquet (dados brutos)
    parquet_blobs = [b.name for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".parquet")]
    log.info(f"Encontrados {len(parquet_blobs)} arquivos parquet brutos.")
    
    # 2. Listar arquivos JSONL (Q&As já gerados)
    jsonl_blobs = [b.name for b in bucket.list_blobs(prefix=out_prefix) if b.name.endswith(".jsonl")]
    log.info(f"Encontrados {len(jsonl_blobs)} arquivos JSONL de Q&A gerados.")
    
    # Mapear nome base do JSONL para o blob path completo
    # Ex: arxiv_pt_20260523_020709_e2316972_chunk_1.jsonl -> gs://...
    jsonl_map = {}
    for jb in jsonl_blobs:
        base = os.path.basename(jb)
        jsonl_map[base] = jb
        
    if limit_files:
        parquet_blobs = parquet_blobs[:limit_files]
        log.info(f"Limitando a importação de parquets a {len(parquet_blobs)} arquivos.")
        
    db_conn = get_db_connection()
    redis_client = redis.Redis.from_url(REDIS_URL)
    
    total_raw_inserted = 0
    total_corpus_inserted = 0
    total_qa_inserted = 0
    
    try:
        for idx, pb_name in enumerate(parquet_blobs):
            base_name = os.path.basename(pb_name)
            expected_jsonl_base = base_name.replace(".parquet", ".jsonl")
            
            has_qa = expected_jsonl_base in jsonl_map
            log.info(f"[{idx+1}/{len(parquet_blobs)}] Processando: {base_name} (Possui Q&A gerado: {has_qa})")
            
            local_parquet = f"/tmp/{base_name}"
            local_jsonl = f"/tmp/{expected_jsonl_base}"
            
            try:
                # Baixar Parquet
                bucket.blob(pb_name).download_to_filename(local_parquet)
                df = pl.read_parquet(local_parquet)
                
                # Identificar spider
                spider_name = "unknown"
                for s in ["wikipedia_pt", "arxiv_pt", "gutenberg_pt", "scielo_pt", "bdtd_pt", "bolema_pt", "remat_pt"]:
                    if s in base_name:
                        spider_name = s
                        break
                
                # Mapear URLs para corpus_id
                url_to_corpus_id = {}
                raw_rows = []
                corpus_rows = []
                
                for row in df.iter_rows(named=True):
                    title = clean_val(row.get("title", ""))
                    text = clean_val(row.get("text", ""))
                    url = clean_val(row.get("url", ""))
                    language = clean_val(row.get("language", "pt"))
                    extracted_at = row.get("extracted_at")
                    char_count = row.get("char_count", len(text))
                    word_count = row.get("word_count", len(text.split()))
                    
                    if not url:
                        url = f"generated_url_{uuid.uuid4().hex}"
                        
                    raw_id = str(uuid.uuid4())
                    corpus_id = str(uuid.uuid4())
                    
                    url_to_corpus_id[url] = corpus_id
                    
                    # Se já tem QA, marcar no raw_corpus como processado para clean e qa
                    p_clean = True if has_qa else False
                    p_qa = True if has_qa else False
                    
                    raw_rows.append((
                        raw_id, title, text, url, language, spider_name, extracted_at, char_count, word_count, p_clean, p_qa
                    ))
                    
                    if has_qa:
                        # Se já tem QA, inserimos o corpus diretamente na tabela corpus
                        meta = {
                            "raw_id": raw_id,
                            "source_file": pb_name,
                            "char_count": char_count
                        }
                        corpus_rows.append((
                            corpus_id, title, text, url, language, extracted_at, None, json.dumps(meta)
                        ))
                
                # 1. Inserir no raw_corpus
                new_raw_ids = []
                with db_conn.cursor() as cur:
                    query_raw = """
                    INSERT INTO raw_corpus (id, title, text, url, language, spider_name, extracted_at, char_count, word_count, processed_clean, processed_qa)
                    VALUES %s
                    ON CONFLICT (url) DO NOTHING
                    RETURNING id;
                    """
                    raw_inserted = execute_values(
                        cur, query_raw, raw_rows, page_size=100, fetch=True
                    )
                    if raw_inserted:
                        new_raw_ids = [r[0] for r in raw_inserted]
                db_conn.commit()
                total_raw_inserted += len(new_raw_ids)
                
                # 2. Se possuir Q&As gerados, importar
                if has_qa:
                    # Inserir na tabela corpus
                    with db_conn.cursor() as cur:
                        query_corpus = """
                        INSERT INTO corpus (id, title, text, url, language, extracted_at, embedding, metadata)
                        VALUES %s
                        ON CONFLICT (id) DO NOTHING;
                        """
                        execute_values(cur, query_corpus, corpus_rows, page_size=100)
                    db_conn.commit()
                    total_corpus_inserted += len(corpus_rows)
                    
                    # Baixar e importar JSONL
                    jsonl_blob_path = jsonl_map[expected_jsonl_base]
                    bucket.blob(jsonl_blob_path).download_to_filename(local_jsonl)
                    
                    qa_rows = []
                    with open(local_jsonl, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            qa_item = json.loads(line)
                            question = clean_val(qa_item.get("question"))
                            reasoning = clean_val(qa_item.get("reasoning", ""))
                            answer = clean_val(qa_item.get("answer"))
                            model = clean_val(qa_item.get("model", "unknown"))
                            source = clean_val(qa_item.get("source", ""))
                            
                            # Achar corpus_id pelo source (url)
                            c_id = url_to_corpus_id.get(source)
                            if not c_id:
                                # Tenta por correspondência parcial ou gera um novo uuid
                                c_id = list(url_to_corpus_id.values())[0] if url_to_corpus_id else None
                                
                            meta_qa = {
                                "corpus_id": c_id,
                                "source_file": jsonl_blob_path
                            }
                            
                            qa_id = str(uuid.uuid4())
                            qa_rows.append((
                                qa_id, question, reasoning, answer, model, source, None, json.dumps(meta_qa)
                            ))
                            
                    if qa_rows:
                        with db_conn.cursor() as cur:
                            query_qa = """
                            INSERT INTO qa_dataset (id, question, reasoning, answer, model, source, embedding, metadata)
                            VALUES %s
                            ON CONFLICT (id) DO NOTHING;
                            """
                            execute_values(cur, query_qa, qa_rows, page_size=100)
                        db_conn.commit()
                        total_qa_inserted += len(qa_rows)
                        
                    # Enfileirar o corpus final no embedder_queue (para gerar embeddings do corpus limpo)
                    if push_to_valkey:
                        for crow in corpus_rows:
                            embedder_payload = {
                                "id": crow[0],
                                "database": "ai_labs",
                                "table": "corpus",
                                "source_column": "text",
                                "target_column": "embedding"
                            }
                            redis_client.lpush("embedder_queue", json.dumps(embedder_payload))
                            
                    log.info(f"✅ Arquivo {base_name} importado: {len(corpus_rows)} corpus, {len(qa_rows)} Q&As persistidos diretamente.")
                    
                else:
                    # Caso NÃO possua Q&As, enfileirar IDs no raw_corpus_queue
                    if push_to_valkey and new_raw_ids:
                        log.info(f"Enfileirando {len(new_raw_ids)} novos IDs no Valkey (raw_corpus_queue)...")
                        for nid in new_raw_ids:
                            redis_client.lpush("raw_corpus_queue", json.dumps({"id": str(nid)}))
                            
                    log.info(f"✅ Arquivo {base_name} importado como BRUTO (pendente de limpeza e QA).")
                    
            except Exception as e:
                log.error(f"Erro ao processar arquivo {base_name}: {e}")
                db_conn.rollback()
            finally:
                for fpath in (local_parquet, local_jsonl):
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        
    finally:
        db_conn.close()
        
    log.info("Smart Backfill concluído com sucesso!")
    log.info(f"Total raw_corpus inseridos: {total_raw_inserted}")
    log.info(f"Total corpus inseridos: {total_corpus_inserted}")
    log.info(f"Total qa_dataset inseridos: {total_qa_inserted}")
    return total_raw_inserted, total_corpus_inserted, total_qa_inserted

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_backfill(limit_files=limit)
