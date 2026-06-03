import os
import json
import logging
import time
import uuid
import psycopg2
from psycopg2.extras import execute_values
import redis
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

class QuotaExceededException(Exception):
    """Exceção lançada quando o limite de RPM (minuto) é atingido."""
    pass

class DailyQuotaExceededException(Exception):
    """Exceção lançada quando o limite de RPD (dia) é atingido."""
    pass

def get_db_connection():
    params = {
        "host": os.environ.get("PG_HOST", "postgres.morescotech.com.br"),
        "port": int(os.environ.get("PG_PORT", 5432)),
        "user": os.environ.get("PG_USER", "yugabyte"),
        "password": os.environ.get("PG_PASSWORD", "YugabytePass2026"),
        "database": os.environ.get("PG_DATABASE", "ai_labs"),
        "sslmode": "disable"
    }
    try:
        return psycopg2.connect(**params, load_balance=True)
    except (TypeError, psycopg2.Error):
        return psycopg2.connect(**params)


PAID_KEY = os.environ.get("GEMINI_API_KEY_PAID") or os.environ.get("GEMINI_API_KEY")
FREE_KEY = os.environ.get("GEMINI_API_KEY_FREE")
PAID_MODEL = os.environ.get("GEMINI_PAID_MODEL", "gemini-2.5-flash-lite")

_EXCLUDE_PATTERNS = [
    "tts", "image", "robotics", "computer-use", "clip", "lyria",
    "nano-banana", "antigravity", "deep-research",
]

# Modelo Gemma 4 26B - garantir que não seja excluído
_GEMMA_4_26B_PATTERNS = ["gemma-4-26b", "gemma4-26b"]

_POOL_LOCK = __import__("threading").Lock()
_POOL: list[dict] = []  # [{key, model_id, key_name, cooldown_until}]
_POOL_COUNTER = 0
_POOL_INITIALIZED = False
POOL_COOLDOWN_SECONDS = 60.0

def _discover_models(api_key: str) -> list[str]:
    import urllib.request as _urlreq
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=200"
        with _urlreq.urlopen(_urlreq.Request(url), timeout=10.0) as r:
            data = json.loads(r.read().decode())
        result = []
        for m in data.get("models", []):
            name = m.get("name", "").replace("models/", "")
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            name_lower = name.lower()
            # Garantir que modelos Gemma 4 26B não sejam excluídos
            if any(pat in name_lower for pat in _GEMMA_4_26B_PATTERNS):
                log.info("[Pool] 🎯 Modelo priorizado: %s", name)
                result.append(name)
                continue
            if any(pat in name_lower for pat in _EXCLUDE_PATTERNS):
                continue
            result.append(name)
        log.info("[Pool] %d modelos descobertos: %s", len(result), result)
        return result
    except Exception as e:
        log.warning("[Pool] Falha ao descobrir modelos: %s", e)
        return []

def _probe_model(api_key: str, model_id: str) -> bool:
    try:
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model_id)
        probe_timeout = 60.0 if "gemma" in model_id.lower() else 10.0
        m.generate_content("Hi", request_options={"timeout": probe_timeout})
        log.info("[Pool] ✅ %s: cota OK", model_id)
        return True
    except Exception as e:
        err = str(e).lower()
        if "429" in err or "quota" in err or "resource_exhausted" in err:
            log.warning("[Pool] ⚠️  %s: sem cota", model_id)
        else:
            log.warning("[Pool] ❌ %s: erro — %s", model_id, str(e)[:80])
        return False

def _init_pool():
    global _POOL, _POOL_INITIALIZED
    with _POOL_LOCK:
        if _POOL_INITIALIZED:
            return
        entries = []
        if FREE_KEY:
            for mid in _discover_models(FREE_KEY):
                if _probe_model(FREE_KEY, mid):
                    entries.append({"key": FREE_KEY, "model_id": mid,
                                    "key_name": "FREE", "cooldown_until": 0.0})

        if not entries:
            log.error("[Pool] Nenhum modelo disponível após probe!")
        else:
            log.info("[Pool] Pool final: %d modelos — %s",
                     len(entries), [e["model_id"] for e in entries])
        _POOL = entries
        _POOL_INITIALIZED = True

def _get_next_pool_entry() -> dict | None:
    global _POOL_COUNTER
    with _POOL_LOCK:
        now = time.time()
        n = len(_POOL)
        if n == 0:
            return None
        for _ in range(n):
            idx = _POOL_COUNTER % n
            _POOL_COUNTER += 1
            entry = _POOL[idx]
            if now > entry["cooldown_until"]:
                return entry
        return None

def _mark_pool_cooldown(entry: dict):
    with _POOL_LOCK:
        entry["cooldown_until"] = time.time() + POOL_COOLDOWN_SECONDS
    log.warning("[Pool] 🔴 %s (%s) em cooldown %ds.",
                entry["model_id"], entry["key_name"], int(POOL_COOLDOWN_SECONDS))

def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "quota" in s or "resource_exhausted" in s

def _is_daily_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "daily" in s or "day" in s or "user rate limit exceeded" in s

def clean_and_extract(text):
    _init_pool()
    prompt = f"""Você é um especialista em extração e limpeza de dados de texto.
Sua tarefa é analisar o seguinte texto bruto (extraído da internet) e retornar um JSON com o texto limpo e metadados.

Regras de Limpeza:
1. Remova links de navegação, cabeçalhos de site, rodapés, "leia mais", termos de uso, etc.
2. Remova informações pessoais sensíveis (PII) se houver.
3. Se o texto principal não for sobre o conteúdo esperado ou for apenas um menu, retorne null.

Extraia também os seguintes metadados:
- topic: O tópico principal do texto (em pt-BR)
- difficulty: Nível de dificuldade da leitura (basico, intermediario, avancado)
- domain: A área de conhecimento (ex: Matemática, Ciência da Computação, Filosofia)
- language_quality: Uma pontuação de 1 a 10 avaliando a qualidade e coerência do texto.

Responda APENAS com um objeto JSON no formato exato:
{{
    "cleaned_text": "texto limpo aqui...",
    "topic": "tópico...",
    "difficulty": "intermediario",
    "domain": "dominio...",
    "language_quality": 8
}}

TEXTO BRUTO:
{text[:4000]}
"""
    n = len(_POOL)
    tried = set()

    for _ in range(n + 2):
        entry = _get_next_pool_entry()
        if entry is None:
            with _POOL_LOCK:
                now = time.time()
                if _POOL:
                    # filter out models with indefinitely long cooldowns (daily quota limit)
                    active_cooldowns = [e["cooldown_until"] - now for e in _POOL if e["cooldown_until"] - now < 3000]
                    if active_cooldowns:
                        wait = max(1.0, min(active_cooldowns))
                    else:
                        log.error("[Pool] Todos os modelos ativos esgotaram cota diária ou de minuto.")
                        raise DailyQuotaExceededException("Todos os modelos esgotaram a cota diária.")
                else:
                    log.error("[Pool] Pool vazio — sem modelos disponíveis.")
                    raise DailyQuotaExceededException("Pool vazio — sem modelos disponíveis.")
            log.warning("[Pool] Todos em cooldown de minuto. Aguardando %.0fs...", wait)
            time.sleep(wait)
            entry = _get_next_pool_entry()
            if entry is None:
                log.error("[Pool] Ainda sem modelos após espera.")
                return None

        mid = entry["model_id"]
        if mid in tried:
            continue
        tried.add(mid)

        log.info("[Pool] 🟢 Usando %s (%s)", mid, entry["key_name"])
        try:
            genai.configure(api_key=entry["key"])
            model = genai.GenerativeModel(mid)
            # Ajuste de timeout específico para modelos Gemma
            # Gemma 4 26B e modelos grandes precisam de mais tempo
            if "gemma-4-26b" in mid.lower():
                timeout = 300.0  # 5 minutos para modelos muito grandes
            elif "gemma" in mid.lower():
                timeout = 180.0  # 3 minutos para outros modelos Gemma
            else:
                timeout = 30.0  # 30 segundos para outros modelos
            
            gen_config = None
            if "gemma" not in mid.lower():
                gen_config = genai.types.GenerationConfig(response_mime_type="application/json")
                
            response = model.generate_content(
                prompt,
                generation_config=gen_config,
                request_options={"timeout": timeout}
            )
            
            resp_text = response.text.strip()
            # If JSON parsing fails directly, try extracting JSON from markdown fences
            try:
                return json.loads(resp_text)
            except json.JSONDecodeError:
                if "```json" in resp_text:
                    resp_text = resp_text.split("```json")[1].split("```")[0].strip()
                elif "```" in resp_text:
                    resp_text = resp_text.split("```")[1].split("```")[0].strip()
                return json.loads(resp_text)
        except Exception as e:
            # If it's a JSON Decode Error, we shouldn't put the model in cooldown
            if isinstance(e, json.JSONDecodeError):
                log.error("[Pool] Falha ao decodificar JSON gerado por %s: %s", mid, e)
                # Try next model in pool instead of returning None directly
                continue
            if _is_rate_limit(e):
                if _is_daily_limit(e):
                    # Mark model with daily limit (long cooldown of 24h)
                    with _POOL_LOCK:
                        entry["cooldown_until"] = time.time() + 86400.0
                    log.warning("[Pool] 🔴 Cota diária excedida para %s (%s). Desativando por 24h.", mid, entry["key_name"])
                    continue
                else:
                    _mark_pool_cooldown(entry)
                    continue
            log.error("[Pool] Erro não-recuperável em %s: %s", mid, e)
            return None

    raise QuotaExceededException("Todos os modelos falharam ou estão em cooldown.")

def process_item(redis_client, queue_name, raw_id):
    db_conn = get_db_connection()
    try:
        # 1. Obter registro do raw_corpus
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT title, text, url, language, extracted_at FROM raw_corpus WHERE id = %s;",
                (raw_id,)
            )
            row = cur.fetchone()
            
        if not row:
            log.warning(f"Item com ID {raw_id} não encontrado na tabela raw_corpus.")
            return True
            
        title, raw_text, url, language, extracted_at = row
        
        if len(raw_text) < 100:
            log.info(f"Texto bruto do ID {raw_id} muito curto ({len(raw_text)} chars). Pulando limpeza.")
            with db_conn.cursor() as cur:
                cur.execute("UPDATE raw_corpus SET processed_clean = TRUE WHERE id = %s;", (raw_id,))
            db_conn.commit()
            return True
            
        t0 = time.time()
        log.info(f"Limpando corpus {raw_id} (tamanho: {len(raw_text)} chars)...")
        
        # 2. Chamar limpeza Gemma
        result_json = clean_and_extract(raw_text)
        elapsed = time.time() - t0
        
        log.info(f"Corpus {raw_id} limpo em {elapsed:.2f}s.")
        
        if result_json and result_json.get("cleaned_text"):
            cleaned_text = result_json["cleaned_text"]
            # Combinar parte limpa com o restante original se for muito grande
            if len(raw_text) > 4000:
                cleaned_text += raw_text[4000:]
                
            meta = {
                "raw_id": raw_id,
                "topic": result_json.get("topic"),
                "difficulty": result_json.get("difficulty"),
                "domain": result_json.get("domain"),
                "language_quality": result_json.get("language_quality"),
                "char_count": len(cleaned_text)
            }
            
            new_corpus_id = str(uuid.uuid4())
            
            # 3. Salvar no banco
            with db_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO corpus (id, title, text, url, language, extracted_at, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (new_corpus_id, title, cleaned_text, url, language, extracted_at, None, json.dumps(meta))
                )
                cur.execute(
                    "UPDATE raw_corpus SET processed_clean = TRUE WHERE id = %s;",
                    (raw_id,)
                )
            db_conn.commit()
            
            # 4. Enfileirar no Valkey para Embedder
            embedder_payload = {
                "id": new_corpus_id,
                "database": "ai_labs",
                "table": "corpus",
                "source_column": "text",
                "target_column": "embedding"
            }
            redis_client.lpush(queue_name, json.dumps(embedder_payload))
            
            # 5. Enfileirar no Valkey para QA Generator
            qa_payload = {
                "id": new_corpus_id,
                "raw_id": raw_id
            }
            redis_client.lpush("qa_queue", json.dumps(qa_payload))
            
            log.info(f"✅ Registro limpo {new_corpus_id} inserido e enfileirado para embedding e QA.")
        else:
            log.warning(f"Limpeza de {raw_id} retornou resultado nulo. Marcando como processado sem gravação.")
            with db_conn.cursor() as cur:
                cur.execute("UPDATE raw_corpus SET processed_clean = TRUE WHERE id = %s;", (raw_id,))
            db_conn.commit()
            
        # Respeitar cota ativa de 15 RPM
        sleep_time = max(0.1, 4.1 - elapsed)
        time.sleep(sleep_time)
        return True
        
    except DailyQuotaExceededException as e:
        log.warning(f"Daily Quota Exceeded: {e}")
        raise
    except QuotaExceededException as e:
        log.warning(f"Quota Exceeded: {e}")
        raise
    except Exception as e:
        log.error(f"Erro ao processar item {raw_id}: {e}")
        db_conn.rollback()
        return False
    finally:
        db_conn.close()

def main():
    redis_url = os.environ.get("REDIS_URL", "redis://valkey-primary.default.svc.cluster.local:6379")
    queue_name = os.environ.get("EMBEDDER_QUEUE_NAME", "embedder_queue")
    
    log.info(f"Iniciando Daemon do Corpus Cleaner. Conectando ao Redis em {redis_url}...")
    r = redis.Redis.from_url(redis_url)
    r.ping()
    log.info("Conexão com Valkey estabelecida. Aguardando tarefas na fila 'raw_corpus_queue'...")
    
    while True:
        try:
            # 1. Buscar tarefa na fila
            result = r.brpop("raw_corpus_queue", timeout=5)
            if not result:
                if r.llen("raw_corpus_queue") == 0:
                    log.info("Fila 'raw_corpus_queue' vazia. Encerrando o processamento do dia com sucesso.")
                    import sys
                    sys.exit(0)
                continue
                
            _, payload_json = result
            payload = json.loads(payload_json)
            raw_id = payload.get("id")
            
            if not raw_id:
                log.warning(f"Payload inválido recebido no raw_corpus_queue: {payload_json}")
                continue
                
            # 2. Processar com resiliência a limites de cota
            processed = False
            while not processed:
                try:
                    process_item(r, queue_name, raw_id)
                    processed = True
                except DailyQuotaExceededException:
                    log.warning("Cota diária atingida. Devolvendo item para a fila e encerrando o pod...")
                    r.lpush("raw_corpus_queue", payload_json)
                    import sys
                    sys.exit(0)
                except QuotaExceededException:
                    # Cota de minutos: hibernar por 60s
                    log.warning("Cota de minuto atingida. Aguardando 60 segundos...")
                    time.sleep(60)
                    
        except redis.exceptions.ConnectionError:
            log.warning("Conexão com Valkey perdida no Cleaner Daemon. Tentando reconectar...")
            time.sleep(2)
        except Exception as e:
            log.error(f"Erro no loop do Cleaner Daemon: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
