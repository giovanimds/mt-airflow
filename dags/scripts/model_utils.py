import json
import urllib.request
import os

def get_available_models() -> list[str]:
    """Fetches available models dynamically to populate the dropdown with 'provider: model' format."""
    models = []
    
    # 1. vLLM
    vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
    try:
        req = urllib.request.Request(f"{vllm_url}/v1/models")
        with urllib.request.urlopen(req, timeout=1.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                for m in data.get("data", []):
                    models.append(f"vllm: {m.get('id')}")
    except Exception:
        pass

    # 2. Mistral API
    mistral_api_key = os.environ.get("MISTRAL_API_KEY")
    if mistral_api_key:
        try:
            req = urllib.request.Request("https://api.mistral.ai/v1/models", headers={"Authorization": f"Bearer {mistral_api_key}"})
            with urllib.request.urlopen(req, timeout=1.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    for m in data.get("data", []):
                        models.append(f"mistral: {m.get('id')}")
        except Exception:
            models.extend([f"mistral: {m}" for m in ["mistral-large-latest", "mistral-small-latest", "ministral-3b-latest", "ministral-8b-latest"]])
    else:
        models.extend([f"mistral: {m}" for m in ["mistral-large-latest", "mistral-small-latest", "ministral-3b-latest", "ministral-8b-latest"]])

    # 3. DeepSeek API
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        try:
            req = urllib.request.Request("https://api.deepseek.com/models", headers={"Authorization": f"Bearer {deepseek_api_key}"})
            with urllib.request.urlopen(req, timeout=1.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    for m in data.get("data", []):
                        models.append(f"deepseek: {m.get('id')}")
        except Exception:
            models.extend([f"deepseek: {m}" for m in ["deepseek-chat", "deepseek-reasoner"]])
    else:
         models.extend([f"deepseek: {m}" for m in ["deepseek-chat", "deepseek-reasoner"]])

    # 4. Gemini API
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_api_key:
        try:
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_api_key}")
            with urllib.request.urlopen(req, timeout=1.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    for m in data.get("models", []):
                        if "generateContent" in m.get("supportedGenerationMethods", []):
                            name = m.get("name", "").replace("models/", "")
                            models.append(f"gemini: {name}")
        except Exception:
            models.extend([f"gemini: {m}" for m in ["gemini-2.0-flash", "gemini-2.0-pro-exp-02-05", "gemini-1.5-flash", "gemini-1.5-pro"]])
    else:
        models.extend([f"gemini: {m}" for m in ["gemini-2.0-flash", "gemini-2.0-pro-exp-02-05", "gemini-1.5-flash", "gemini-1.5-pro"]])

    # Fallback/defaults if not found dynamically
    default_models = ["vllm: Meta-Llama-3.1-8B-Instruct", "vllm: granite4.1:8b", "vllm: llama3.2:3b"]
    for d in default_models:
        if d not in models:
            models.insert(0, d)

    models.append("Customizado (digitar no campo abaixo)")
    
    # Remove duplicates preserving order
    return list(dict.fromkeys(models))
