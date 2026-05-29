import os
import json
import logging
import requests
import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
REDIS_URL = os.environ.get("REDIS_URL", "redis://valkey-primary.default.svc.cluster.local:6379")

def generate_topics(main_topic: str, num_terms: int = 15) -> list[str]:
    if not MISTRAL_API_KEY:
        log.error("MISTRAL_API_KEY não configurada no ambiente.")
        raise ValueError("MISTRAL_API_KEY não configurada.")

    log.info(f"Gerando {num_terms} tópicos para o assunto principal: '{main_topic}' usando Mistral...")

    prompt = f"""Você é um curador de dados encarregado de expandir uma base de conhecimento diversificada, fluida e de alta qualidade em português (pt-BR).
Com base no assunto principal "{main_topic}", gere exatamente {num_terms} sub-tópicos ou conceitos específicos.
Buscamos cobrir não apenas fatos acadêmicos, mas também aspectos de interação humana, diálogo educado, resolução de dúvidas comuns do dia a dia, e fluidez na comunicação sobre o tema.

Retorne APENAS um objeto JSON válido no formato:
{{
    "search_terms": [
        "termo/assunto 1",
        "termo/assunto 2",
        ...
    ]
}}"""

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }

    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.7
    }

    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        res_data = response.json()
        content = res_data["choices"][0]["message"]["content"]
        
        data = json.loads(content)
        terms = data.get("search_terms", [])
        log.info(f"Mistral gerou {len(terms)} termos com sucesso.")
        return [str(t).strip() for t in terms if str(t).strip()]
    except Exception as e:
        log.error(f"Erro ao chamar API do Mistral ou parsear JSON: {e}")
        raise

def push_topics_to_queue(topics: list[str]) -> tuple[int, int]:
    log.info(f"Conectando ao Valkey em: {REDIS_URL}")
    r = redis.Redis.from_url(REDIS_URL)
    
    # Testar conexão
    r.ping()
    
    added_count = 0
    duplicate_count = 0
    
    for topic in topics:
        # SADD retorna 1 se o elemento foi adicionado, ou 0 se já existia
        is_new = r.sadd("processed_topics_set", topic)
        if is_new:
            # LPUSH na fila de busca
            r.lpush("search_topics_queue", topic)
            added_count += 1
        else:
            duplicate_count += 1
            
    log.info(f"Fila atualizada: {added_count} novos tópicos adicionados, {duplicate_count} duplicados ignorados.")
    return added_count, duplicate_count

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "Inteligência Artificial"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    try:
        terms = generate_topics(topic, n)
        push_topics_to_queue(terms)
    except Exception as e:
        log.error(f"Falha na execução: {e}")
        sys.exit(1)
