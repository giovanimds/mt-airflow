import os
import json
import logging
import time
import traceback
import psycopg2
from psycopg2.extras import execute_batch
from langchain_mistralai import MistralAIEmbeddings
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

def get_db_connection(dbname):
    return psycopg2.connect(
        host=os.environ.get("PG_HOST", "postgres.morescotech.com.br"),
        port=int(os.environ.get("PG_PORT", 5432)),
        user=os.environ.get("PG_USER", "yugabyte"),
        password=os.environ.get("PG_PASSWORD", "YugabytePass2026"),
        database=dbname,
        sslmode="disable"
    )

def process_batch(batch):
    if not batch:
        return

    log.info(f"Processando batch de {len(batch)} itens.")
    
    # Agrupar por banco de dados e tabela para otimizar conexões e queries
    # Estrutura: grouped[db_name][table_name] = [item1, item2, ...]
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
        model="mistral-embed"
    )

    for db_name, tables in grouped.items():
        try:
            with get_db_connection(db_name) as conn:
                with conn.cursor() as cur:
                    for table_name, items in tables.items():
                        # Assumimos que source_column e target_column são consistentes na mesma tabela
                        source_col = items[0].get("source_column", "text")
                        target_col = items[0].get("target_column", "embedding")
                        
                        ids = [item.get("id") for item in items]
                        id_placeholders = ",".join(["%s"] * len(ids))
                        
                        # Buscar os textos que precisam de embedding
                        cur.execute(
                            f"SELECT id, {source_col} FROM {table_name} WHERE id IN ({id_placeholders});",
                            tuple(ids)
                        )
                        rows = cur.fetchall()
                        
                        if not rows:
                            continue
                            
                        # Extrair os textos e manter a ordem dos IDs
                        texts_to_embed = []
                        valid_ids = []
                        for row_id, text in rows:
                            if text and isinstance(text, str) and len(text.strip()) > 0:
                                # Garantir que o texto não ultrapassa limites absurdos (truncando se necessário)
                                # Mistral Embed suporta contextos grandes, mas é bom prevenir
                                texts_to_embed.append(text[:20000]) 
                                valid_ids.append(row_id)
                        
                        if not texts_to_embed:
                            continue
                            
                        log.info(f"Gerando embeddings para {len(texts_to_embed)} textos da tabela {db_name}.{table_name}...")
                        embeddings = embedder.embed_documents(texts_to_embed)
                        
                        # Fazer o UPDATE no YugabyteDB
                        update_query = f"UPDATE {table_name} SET {target_col} = %s WHERE id = %s;"
                        update_data = [(json.dumps(emb), row_id) for emb, row_id in zip(embeddings, valid_ids)]
                        
                        execute_batch(cur, update_query, update_data)
                        conn.commit()
                        log.info(f"✅ {len(update_data)} registros atualizados na tabela {db_name}.{table_name}.")
                        
        except Exception as e:
            log.error(f"Erro ao processar batch para o DB {db_name}: {e}")
            traceback.print_exc()
            # Retornar itens para a fila em caso de erro?
            # Para evitar loops infinitos, apenas logamos, mas idealmente teríamos uma DLQ.

def main():
    log.info(f"Iniciando Universal Embedder. Conectando ao Redis em {REDIS_URL}...")
    r = redis.Redis.from_url(REDIS_URL, socket_timeout=POLL_TIMEOUT_SEC + 5)
    
    # Testar conexão
    r.ping()
    log.info("Conexão com Redis/Valkey estabelecida com sucesso.")
    
    while True:
        try:
            batch = []
            # Pegar o primeiro item com bloqueio (BRPOP)
            result = r.brpop(QUEUE_NAME, timeout=POLL_TIMEOUT_SEC)
            
            if result:
                _, item_json = result
                batch.append(json.loads(item_json))
                
                # Pegar o restante do batch sem bloqueio (RPOP)
                while len(batch) < BATCH_SIZE:
                    item_json = r.rpop(QUEUE_NAME)
                    if item_json:
                        batch.append(json.loads(item_json))
                    else:
                        break
                
                process_batch(batch)
                
        except redis.exceptions.TimeoutError:
            # Timeout normal do BRPOP quando a fila está vazia
            continue
        except redis.exceptions.ConnectionError:
            log.warning("Conexão com Valkey perdida. Tentando reconectar...")
            time.sleep(2)
        except Exception as e:
            log.error(f"Erro no loop principal: {e}")
            time.sleep(5) # Aguardar um pouco antes de tentar novamente em caso de erro crítico

if __name__ == "__main__":
    main()
