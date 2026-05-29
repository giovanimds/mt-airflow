import json
import urllib.request
import os
import logging
import psycopg2

log = logging.getLogger(__name__)

def get_db_connection():
    """Gets a connection to YugabyteDB using explicit credentials to avoid Airflow's env masking."""
    # Try custom env var first, then fallback to known cluster defaults
    conn_uri = os.environ.get("CUSTOM_DB_CONN")
    if conn_uri:
        try:
            return psycopg2.connect(conn_uri.replace("postgresql://", "postgres://"))
        except Exception: pass

    # Cluster defaults (based on established credentials)
    try:
        return psycopg2.connect(
            host="postgres.default.svc.cluster.local",
            port=5432,
            user="yugabyte",
            password="YugabytePass2026",
            database="airflow",
            sslmode="disable"
        )
    except Exception as e:
        log.warning("Failed to connect to YugabyteDB (Cluster Defaults): %s", e)
        return None

def get_available_models() -> list[str]:
    """Fetches available models from YugabyteDB to avoid dynamic API calls during parsing."""
    models = []
    
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT provider, model_id FROM available_models ORDER BY provider, model_id;")
                rows = cur.fetchall()
                for provider, model_id in rows:
                    models.append(f"{provider}: {model_id}")
            conn.close()
        except Exception as e:
            log.warning("Failed to fetch models from DB: %s", e)

    # 2. If DB is empty or failed, return a minimal static list
    if not models:
        models = [
            "mistral: mistral-pool",
            "vllm: Meta-Llama-3.1-8B-Instruct",
            "gemini: gemini-2.0-flash",
            "mistral: mistral-large-latest",
            "deepseek: deepseek-chat"
        ]

        ]

    # 3. Sort models but keep preferred ones at the top
    preferred = ["mistral: mistral-pool", "vllm: Meta-Llama-3.1-8B-Instruct"]
    final_list = [m for m in preferred if m in models]
    final_list += sorted([m for m in models if m not in preferred])

    final_list.append("Customizado (digitar no campo abaixo)")
    return list(dict.fromkeys(final_list))

def fetch_models_from_apis() -> list[tuple[str, str]]:
    """
    Dynamic API fetching logic. Returns a list of (provider, model_id) tuples.
    """
    results = []
    
    # 1. vLLM
    vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
    try:
        req = urllib.request.Request(f"{vllm_url}/v1/models")
        with urllib.request.urlopen(req, timeout=5.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                for m in data.get("data", []):
                    results.append(("vllm", m.get('id')))
    except Exception as e:
        log.warning("vLLM fetch failed: %s", e)

    # 2. Mistral
    mistral_api_key = os.environ.get("MISTRAL_API_KEY")
    if mistral_api_key:
        results.append(("mistral", "mistral-pool"))
        try:
            req = urllib.request.Request("https://api.mistral.ai/v1/models", headers={"Authorization": f"Bearer {mistral_api_key}"})
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    for m in data.get("data", []):
                        results.append(("mistral", m.get('id')))
        except Exception:
            for m in ["mistral-large-latest", "mistral-small-latest", "ministral-3b-latest", "ministral-8b-latest"]:
                results.append(("mistral", m))
    
    # 3. DeepSeek
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        try:
            req = urllib.request.Request("https://api.deepseek.com/models", headers={"Authorization": f"Bearer {deepseek_api_key}"})
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    for m in data.get("data", []):
                        results.append(("deepseek", m.get('id')))
        except Exception:
            for m in ["deepseek-chat", "deepseek-reasoner"]:
                results.append(("deepseek", m))

    # 4. Gemini
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_api_key:
        try:
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_api_key}")
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    for m in data.get("models", []):
                        if "generateContent" in m.get("supportedGenerationMethods", []):
                            name = m.get("name", "").replace("models/", "")
                            results.append(("gemini", name))
        except Exception:
            for m in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
                results.append(("gemini", m))

    return results
