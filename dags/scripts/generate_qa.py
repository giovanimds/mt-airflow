import os
import json
import logging
import traceback
import polars as pl
import urllib.request
import time
import threading

from google.cloud import storage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini Pool LLM — descoberta dinâmica + round-robin com cooldown por modelo
# ---------------------------------------------------------------------------
_GEMINI_POOL_LOCK = threading.Lock()

# Modelos a excluir do pool (TTS, imagem, especialistas, robotics, etc.)
_GEMINI_EXCLUDE_PATTERNS = [
    "tts", "image", "robotics", "computer-use", "clip", "lyria",
    "nano-banana", "antigravity", "deep-research",
]

def _discover_gemini_models(api_key: str) -> list[str]:
    """
    Chama /v1beta/models e retorna IDs dos modelos que suportam generateContent,
    excluindo modelos de TTS, imagem e especialistas.
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=200"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode())
        models = data.get("models", [])
        result = []
        for m in models:
            name = m.get("name", "").replace("models/", "")
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            # Filtra modelos inúteis para o pipeline de texto
            if any(pat in name.lower() for pat in _GEMINI_EXCLUDE_PATTERNS):
                continue
            result.append(name)
        log.info("[GeminiPool] %d modelos descobertos via API: %s", len(result), result)
        return result
    except Exception as e:
        log.warning("[GeminiPool] Falha ao descobrir modelos: %s", e)
        return []

def _probe_gemini_model(api_key: str, model_id: str) -> bool:
    """
    Faz uma chamada mínima real para confirmar que o modelo tem cota disponível.
    Retorna True se respondeu com sucesso.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.0,
            max_retries=0,
            max_output_tokens=16,
            google_api_key=api_key,
        )
        llm.invoke("Hi")
        log.info("[GeminiPool] ✅ %s: cota OK", model_id)
        return True
    except Exception as e:
        err = str(e).lower()
        if "429" in err or "quota" in err or "resource_exhausted" in err:
            log.warning("[GeminiPool] ⚠️  %s: sem cota (rate-limit)", model_id)
        else:
            log.warning("[GeminiPool] ❌ %s: erro (%s)", model_id, str(e)[:80])
        return False


class GeminiPoolLLM:
    """
    Pool dinâmico de modelos Gemini com:
    - Descoberta automática via /v1beta/models na startup
    - Probe de cota real para cada modelo
    - Round-robin thread-safe entre modelos ativos
    - Cooldown por modelo (60s) ao bater rate-limit — não descarta, apenas pausa
    - Fallback para PAID key como última opção
    - `with_structured_output` propaga o pool para wrappers estruturados
    """
    COOLDOWN_SECONDS = 60.0

    def __init__(self, entries: list[dict], is_structured: bool = False):
        """
        entries: lista de dicts com keys 'llm', 'model_id', 'key_name', 'cooldown_until'
        """
        self._entries = entries  # [{"llm": ..., "model_id": ..., "key_name": ..., "cooldown_until": 0.0}]
        self._counter = 0
        self._is_structured = is_structured

    @classmethod
    def build(
        cls,
        free_key: str | None,
        paid_key: str | None,
        paid_model: str = "gemini-2.5-flash-lite",
        max_retries: int = 3,
        max_output_tokens: int = 32768,
        probe: bool = True,
    ) -> "GeminiPoolLLM":
        """
        Descobre e sonda todos os modelos disponíveis.
        FREE key  → todos os modelos da API que passam no probe
        PAID key  → modelo pago fixo (fallback de última instância)
        """
        entries = []

        # ---- FREE: descoberta dinâmica ----
        if free_key:
            model_ids = _discover_gemini_models(free_key)
            log.info("[GeminiPool] Sondando %d modelos FREE...", len(model_ids))
            for mid in model_ids:
                ok = _probe_gemini_model(free_key, mid) if probe else True
                if ok:
                    llm = ChatGoogleGenerativeAI(
                        model=mid,
                        temperature=0.5,
                        max_retries=0,
                        max_output_tokens=max_output_tokens,
                        google_api_key=free_key,
                    )
                    entries.append({
                        "llm": llm,
                        "model_id": mid,
                        "key_name": "FREE",
                        "cooldown_until": 0.0,
                    })

        # ---- PAID: modelo fixo, adicionado ao final como fallback ----
        if paid_key:
            paid_llm = ChatGoogleGenerativeAI(
                model=paid_model,
                temperature=0.5,
                max_retries=max_retries,
                max_output_tokens=max_output_tokens,
                google_api_key=paid_key,
            )
            entries.append({
                "llm": paid_llm,
                "model_id": paid_model,
                "key_name": "PAID",
                "cooldown_until": 0.0,
            })
            log.info("[GeminiPool] PAID adicionado ao pool: %s", paid_model)

        if not entries:
            raise ValueError("[GeminiPool] Nenhum modelo Gemini disponível após probe.")

        log.info(
            "[GeminiPool] Pool final: %d modelos — %s",
            len(entries),
            [e["model_id"] for e in entries],
        )
        return cls(entries)

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        s = str(exc).lower()
        return "429" in s or "quota" in s or "resource_exhausted" in s

    def _get_next_available(self) -> dict | None:
        """Round-robin ignorando modelos em cooldown."""
        with _GEMINI_POOL_LOCK:
            now = time.time()
            n = len(self._entries)
            for _ in range(n):
                idx = self._counter % n
                self._counter += 1
                entry = self._entries[idx]
                if now > entry["cooldown_until"]:
                    return entry
            return None  # todos em cooldown

    def _is_daily_limit_error(self, exc: Exception) -> bool:
        s = str(exc).lower()
        return "daily" in s or "day" in s or "user rate limit exceeded" in s

    def _mark_cooldown(self, entry: dict, seconds: float = 60.0):
        with _GEMINI_POOL_LOCK:
            entry["cooldown_until"] = time.time() + seconds
        log.warning(
            "[GeminiPool] 🔴 %s (%s) em cooldown %ds.",
            entry["model_id"], entry["key_name"], int(seconds),
        )

    def _invoke_with_pool(self, method_name: str, *args, **kwargs):
        """
        Tenta cada modelo disponível no pool em round-robin.
        Ao bater rate-limit, coloca modelo em cooldown e tenta o próximo.
        """
        n = len(self._entries)
        tried = set()

        for _ in range(n + 1):  # +1 para dar segunda chance após cooldowns
            entry = self._get_next_available()
            if entry is None:
                # Todos em cooldown — aguarda o menor cooldown e retenta
                with _GEMINI_POOL_LOCK:
                    now = time.time()
                    # Filter out models that have daily quota limits (long cooldowns)
                    active_cooldowns = [e["cooldown_until"] - now for e in self._entries if e["cooldown_until"] - now < 3000]
                    if active_cooldowns:
                        wait = max(1.0, min(active_cooldowns))
                    else:
                        raise RuntimeError("[GeminiPool] Todos os modelos esgotaram a cota diária ou de minuto.")
                log.warning("[GeminiPool] Todos os modelos em cooldown. Aguardando %.0fs...", wait)
                time.sleep(wait)
                entry = self._get_next_available()
                if entry is None:
                    raise RuntimeError("[GeminiPool] Nenhum modelo disponível após espera.")

            model_id = entry["model_id"]
            if model_id in tried:
                continue
            tried.add(model_id)

            log.info("[GeminiPool] 🟢 Usando %s (%s)", model_id, entry["key_name"])
            try:
                method = getattr(entry["llm"], method_name)
                return method(*args, **kwargs)
            except Exception as exc:
                if self._is_rate_limit_error(exc):
                    if self._is_daily_limit_error(exc):
                        # Daily limit -> long cooldown of 24h
                        self._mark_cooldown(entry, 86400.0)
                    else:
                        # Minute limit -> 60s cooldown
                        self._mark_cooldown(entry, 60.0)
                    log.info("[GeminiPool] Tentando próximo modelo do pool...")
                    continue
                raise  # Erro não-recuperável

        raise RuntimeError(f"[GeminiPool] Todos os {n} modelos falharam ou estão em cooldown.")

    def invoke(self, input, config=None, **kwargs):
        return self._invoke_with_pool("invoke", input, config=config, **kwargs)

    def batch(self, inputs, config=None, **kwargs):
        return self._invoke_with_pool("batch", inputs, config=config, **kwargs)

    def with_structured_output(self, schema, **kwargs) -> "GeminiPoolLLM":
        """Propaga structured_output para todos os LLMs do pool."""
        new_entries = []
        for e in self._entries:
            new_entries.append({
                **e,
                "llm": e["llm"].with_structured_output(schema, **kwargs),
            })
        return GeminiPoolLLM(new_entries, is_structured=True)

# ---------------------------------------------------------------------------
# YugabyteDB Connection & Syncing Utilities
# ---------------------------------------------------------------------------
def get_db_connection():
    import psycopg2
    params = {
        "host": os.environ.get("PG_HOST", "postgres.default.svc.cluster.local"),
        "port": int(os.environ.get("PG_PORT", 5432)),
        "user": os.environ.get("PG_USER", "yugabyte"),
        "password": os.environ.get("PG_PASSWORD", "YugabytePass2026"),
        "database": os.environ.get("PG_DATABASE", "ai_labs"),
        "sslmode": os.environ.get("PG_SSLMODE", "disable")
    }
    try:
        return psycopg2.connect(**params, load_balance=True)
    except (TypeError, psycopg2.Error):
        return psycopg2.connect(**params)

def clean_string(s):
    if s is None:
        return None
    if isinstance(s, str):
        return s.replace('\x00', '').replace('\u0000', '')
    return str(s).replace('\x00', '').replace('\u0000', '')

def save_corpus_to_db(conn, df, file_name):
    from psycopg2.extras import execute_values
    log.info("Salvando corpus do arquivo %s no YugabyteDB...", file_name)
    rows_to_insert = []
    for row in df.iter_rows(named=True):
        title = clean_string(row.get("title"))
        text = clean_string(row.get("text"))
        url = clean_string(row.get("url"))
        language = clean_string(row.get("language"))
        extracted_at = row.get("extracted_at")
        char_count = row.get("char_count")
        word_count = row.get("word_count")
        
        meta = {"source_file": file_name}
        
        rows_to_insert.append((
            title,
            text,
            url,
            language,
            extracted_at,
            char_count,
            word_count,
            None, # embedding
            json.dumps(meta) # metadata
        ))
        
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO corpus (title, text, url, language, extracted_at, char_count, word_count, embedding, metadata)
            VALUES %s
            """,
            rows_to_insert
        )
        cur.execute(
            "INSERT INTO imported_files (file_name, file_type) VALUES (%s, %s) ON CONFLICT (file_name) DO NOTHING;",
            (file_name, "corpus")
        )
    conn.commit()
    log.info("Corpus de %s salvo com sucesso no YugabyteDB!", file_name)

def save_qa_to_db(conn, results_jsonl, file_name):
    from psycopg2.extras import execute_values
    if not results_jsonl:
        return
    log.info("Salvando %d Q&As do arquivo %s no YugabyteDB...", len(results_jsonl), file_name)
    rows_to_insert = []
    for item in results_jsonl:
        question = clean_string(item.get("question"))
        reasoning = clean_string(item.get("reasoning"))
        answer = clean_string(item.get("answer"))
        model = clean_string(item.get("model"))
        source = clean_string(item.get("source"))
        
        meta = {"source_file": file_name}
        
        rows_to_insert.append((
            question,
            reasoning,
            answer,
            model,
            source,
            None, # embedding
            json.dumps(meta) # metadata
        ))
        
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO qa_dataset (question, reasoning, answer, model, source, embedding, metadata)
            VALUES %s
            """,
            rows_to_insert
        )
        cur.execute(
            "INSERT INTO imported_files (file_name, file_type) VALUES (%s, %s) ON CONFLICT (file_name) DO NOTHING;",
            (file_name, "qa")
        )
    conn.commit()
    log.info("Q&As de %s salvos com sucesso no YugabyteDB!", file_name)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
abstract_prompt = ChatPromptTemplate.from_template(
    """Você é um assistente especializado em mineração de dados acadêmicos. 
Analise o texto bruto fornecido e extraia o resumo e as palavras-chave.

Ignore completamente endereços, e-mails, instruções de navegação de sites e direitos de acesso.
Não traduza o texto, apenas extraia as informações solicitadas.

Texto Bruto:
{texto}
"""
)

consolidation_prompt = ChatPromptTemplate.from_template(
    """Você é um pesquisador sênior consolidando múltiplos resumos parciais de um mesmo artigo científico.
Sua tarefa é criar um resumo único, coerente e abrangente que capture todas as descobertas principais, metodologias e conclusões mencionadas nos fragmentos abaixo.

Resumos Parciais:
{resumos_parciais}

Regras:
- Remova redundâncias entre os fragmentos.
- Mantenha a terminologia técnica original.
- O resultado final deve ser um resumo fluido e estruturado.
"""
)


meta_prompt = ChatPromptTemplate.from_template(
    """Com base no resumo acadêmico fornecido, extraia os metadados metodológicos solicitados.

Regras:
- Se uma informação não estiver explícita, preencha o campo de forma vazia ou nula.
- Preencha os alvos sempre em pt-BR, mesmo que o texto esteja em outro idioma.
- eh_ponderativo é true se o artigo fizer uma análise crítica ou reflexão profunda sobre um tema.
- eh_exploratorio é true se o artigo apresentar um estudo exploratório, levantamento de dados ou análise de campo.
- eh_especulativo é true se o artigo apresentar hipóteses, conjecturas ou propostas teóricas sem necessariamente ter dados empíricos.
- eh_embasado_em_dados é true se o artigo apresentar dados empíricos, análises estatísticas ou evidências concretas.
- eh_embasado_em_dados é true APENAS se os dados estiverem explicitamente descritos no resumo.
- eh_destilavel_pergunta_resposta é true se o artigo puder ser resumido em um formato de pergunta e resposta direta.
- eh_destilavel_chain_of_thought é true se o artigo puder ser resumido em um formato de cadeia de raciocínio passo a passo.
- eh_desprovido_de_utilidade_pratica é true se o texto estiver incompleto, for um fragmento de rodapé, instrução de navegação, ou não tiver utilidade prática.
- eh_desprovido_de_utilidade_pratica é true se o texto não contiver pelo menos uma conclusão tangível, um método claro ou um dado concreto.

Resumo:
{abstract}
"""
)

qa_prompt = ChatPromptTemplate.from_template(
    """Você é um pesquisador extraindo conhecimento direto de um artigo científico.
    Com base no resumo fornecido, gere exatamente {quantidade} perguntas diretas.

    REGRA CRÍTICA DE CONTEXTO (ANCORAGEM):
    Nenhuma pergunta pode ser genérica. Toda pergunta DEVE citar o contexto específico do estudo para fazer sentido isoladamente.
    - ERRADO: "Qual foi o principal resultado?", "Existe correlação entre x e y segundo o artigo?"
    - CERTO: Perguntas que carregam o contexto necessário para serem compreendidas sem o resumo.
    - Não mencione que é um artigo ou estudo.
    - Não cite nomes ou localidades específicas; abstraia mantendo o contexto.

    Exemplos ruins:
    "Em um artigo sobre ...", "Segundo um estudo recente ...", "De acordo com um artigo científico ..."

    Resumo:
    {abstract}
    """
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# ... (abstract_prompt, meta_prompt, qa_prompt unchanged)

reasoning_prompt = ChatPromptTemplate.from_template(
    """Você é um pesquisador sênior em uma fase de reflexão profunda.
    Sua tarefa é analisar a pergunta abaixo e desenvolver um raciocínio detalhado, fluido e natural.

    REGRAS DO RACIOCÍNIO (Estrutura Markdown):
    - Escreva como um monólogo interno, explorando o problema de forma discursiva.
    - Organize o texto em seções iniciadas por títulos Markdown de nível 2, como: ## Entendendo o problema, ## Explorando alternativas ou ## Aplicando solução.
    - Termine seu raciocínio OBRIGATORIAMENTE com a seção: ## Formulando resposta seguido de um breve resumo do que será a conclusão.
    - PROIBIDO o uso de listas numeradas ou bullet points dentro das seções.
    - Seja prolixo e detalhado na exploração intelectual.

    Pergunta: {pergunta}
    """
)

answering_prompt = ChatPromptTemplate.from_template(
    """Você é um pesquisador sênior fornecendo uma resposta técnica definitiva.
    Com base no raciocínio detalhado abaixo, forneça a resposta final técnica e detalhada para a pergunta original.

    Pergunta: {pergunta}

    Raciocínio prévio (Seções Markdown):
    {reasoning}

    Sua resposta deve focar exclusivamente na conclusão técnica, sendo direta, completa e revisada.

    REGRA CRÍTICA DE FORMATAÇÃO:
    - NÃO inclua títulos ou cabeçalhos Markdown de qualquer nível (ex: #, ##, ###, etc.) no topo da resposta.
    - NÃO inclua prefixos ou rótulos em negrito no início (ex: NÃO comece com '**Resposta:**' ou '**Resposta Técnica Definitiva:**').
    - Inicie a resposta de forma fluida, direta e natural, partindo direto para a explicação ou solução técnica.
    """
)

# ---------------------------------------------------------------------------
# Schemas (Pydantic)
# ---------------------------------------------------------------------------
class AbstractResult(BaseModel):
    abstract: str = Field(description="O resumo do artigo científico isolado de qualquer texto de interface ou rodapé. Vazio se não for encontrado.")
    keywords: list[str] = Field(description="Uma lista com as palavras-chave explícitas no texto.")

class MetaResult(BaseModel):
    publico_alvo: str | None = Field(default=None)
    conclusoes_principais: str | None = Field(default=None)
    eh_ponderativo: bool = Field(default=False)
    eh_exploratorio: bool = Field(default=False)
    eh_especulativo: bool = Field(default=False)
    eh_embasado_em_dados: bool = Field(default=False)
    eh_destilavel_pergunta_resposta: bool = Field(default=False)
    eh_destilavel_chain_of_thought: bool = Field(default=False)
    eh_desprovido_de_utilidade_pratica: bool = Field(default=False)

class QuestionItem(BaseModel):
    pergunta: str = Field(description="Pergunta contextualizada em pt-BR")

class QuestionsList(BaseModel):
    perguntas: list[QuestionItem]

class AnswerResult(BaseModel):
    answer: str = Field(
        description=(
            "Sua conclusão ou resposta final direta e detalhada. "
            "REGRA CRÍTICA: Comece a responder diretamente. NÃO inclua nenhum título, "
            "rótulo (como 'Resposta:') ou cabeçalhos Markdown (como '#', '##', '###' ou '**Título**') no início da resposta."
        )
    )

class QAResult(BaseModel):
    reasoning: str = Field(description="Seu pensamento fluido organizado por seções Markdown (## Título)")
    answer: str = Field(
        description=(
            "Sua conclusão ou resposta final direta e detalhada. "
            "REGRA CRÍTICA: Comece a responder diretamente. NÃO inclua nenhum título, "
            "rótulo (como 'Resposta:') ou cabeçalhos Markdown (como '#', '##', '###' ou '**Título**') no início da resposta."
        )
    )


def get_field(obj, field_name, default=""):
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline(llm):
    chain_perguntas = qa_prompt | llm.with_structured_output(QuestionsList)
    
    # Decoupled components
    thinking_chain = reasoning_prompt | llm
    answering_chain = answering_prompt | llm.with_structured_output(AnswerResult)

    def robust_abstract_extraction(inputs):
        """
        Extrai o resumo de forma robusta. Se o texto for muito grande,
        usa uma estratégia de Map-Reduce para não perder contexto.
        """
        texto = inputs.get("texto", "")
        # Parâmetros de chunking (em caracteres)
        MAX_BLOCK = 160_000 
        CHUNK_SIZE = 120_000
        OVERLAP = 10_000

        if len(texto) <= MAX_BLOCK:
            return (abstract_prompt | llm.with_structured_output(AbstractResult)).invoke(inputs)
        
        log.info("🔍 Artigo grande (%d chars). Iniciando Map-Reduce para consolidação de resumo...", len(texto))
        
        # 1. Map: Gera resumos parciais para cada chunk
        chunks = []
        for i in range(0, len(texto), CHUNK_SIZE - OVERLAP):
            chunks.append(texto[i : i + CHUNK_SIZE])
            if i + CHUNK_SIZE >= len(texto):
                break
        
        log.debug("Dividido em %d chunks para mapeamento.", len(chunks))
        partial_chain = abstract_prompt | llm.with_structured_output(AbstractResult)
        
        # Batch invoke para os chunks (aproveita concorrência se o provider suportar)
        partials = partial_chain.batch([{"texto": c} for c in chunks])
        
        # 2. Reduce: Consolida resumos parciais em um global
        resumos_text = ""
        todas_keywords = set()
        for idx, p in enumerate(partials):
            if isinstance(p, Exception):
                log.warning("Falha no chunk %d do Map-Reduce: %s", idx, p)
                continue
            
            p_abstract = get_field(p, "abstract", "")
            if p_abstract:
                resumos_text += f"\n--- FRAGMENTO {idx+1} ---\n{p_abstract}\n"
            
            p_keywords = get_field(p, "keywords", [])
            if isinstance(p_keywords, list):
                todas_keywords.update(p_keywords)

        if not resumos_text:
            return AbstractResult(abstract="", keywords=[])

        log.debug("Consolidando %d resumos parciais...", len(partials))
        final_abstract = (consolidation_prompt | llm.with_structured_output(AbstractResult)).invoke({
            "resumos_parciais": resumos_text
        })
        
        # Mescla as keywords originais com as consolidadas
        consolidated_keywords = set(get_field(final_abstract, "keywords", []))
        final_abstract.keywords = list(consolidated_keywords | todas_keywords)
        
        return final_abstract

    def gerar_dataset_desacoplado(inputs):
        resultado_perguntas = chain_perguntas.invoke(inputs)
        lista_perguntas = get_field(resultado_perguntas, "perguntas", []) if resultado_perguntas else []

        dataset = []
        for p in lista_perguntas:
            p_text = get_field(p, "pergunta", "")
            if not p_text:
                continue
            
            try:
                # Passo 1: Gerar Raciocínio Fluido (Texto Puro)
                log.debug("Executando Passo 1: Reasoning para '%s'", p_text[:50])
                res_thinking = thinking_chain.invoke({"pergunta": p_text})
                reasoning_text = res_thinking.content if hasattr(res_thinking, 'content') else str(res_thinking)
                
                if not reasoning_text:
                    continue

                # Passo 2: Gerar Resposta Final Estruturada
                log.debug("Executando Passo 2: Answering para '%s'", p_text[:50])
                res_answer = answering_chain.invoke({
                    "pergunta": p_text,
                    "reasoning": reasoning_text
                })
                
                answer_text = get_field(res_answer, "answer", "")
                
                if answer_text:
                    dataset.append({
                        "instruction": p_text, 
                        "response": {
                            "reasoning": reasoning_text,
                            "answer": answer_text
                        }
                    })
            except Exception as exc:
                log.warning("Falha ao processar Q&A individual para '%s': %s", p_text[:30], exc)
                continue
                
        return dataset

    def roteador_de_lixo(inputs):
        fase_1 = inputs.get("fase_1")
        fase_2 = inputs.get("fase_2")
        
        abstract_text = get_field(fase_1, "abstract", "") if fase_1 else ""
        eh_lixo = get_field(fase_2, "eh_desprovido_de_utilidade_pratica", False) or not abstract_text

        if eh_lixo:
            return {
                "fase_1": fase_1,
                "fase_2": fase_2,
                "dataset_instrucoes": None,
                "status_pipeline": "descartado_para_revisao",
            }

        destilacao_input = {"abstract": abstract_text, "quantidade": 3}
        tarefas_paralelas = {
            "fase_1": lambda _: fase_1,
            "fase_2": lambda _: fase_2,
        }

        tarefas_paralelas["dataset_instrucoes"] = (
            (lambda _: destilacao_input) | RunnableLambda(gerar_dataset_desacoplado)
        )

        tarefas_paralelas["status_pipeline"] = lambda _: "processado_com_sucesso"
        return RunnableParallel(**tarefas_paralelas).invoke(inputs)

    return (
        RunnableLambda(robust_abstract_extraction)
        | {
            "fase_1": lambda x: x,
            "fase_2": {"abstract": lambda x: get_field(x, "abstract", "") if x else ""} | meta_prompt | llm.with_structured_output(MetaResult),
        }
        | RunnableLambda(roteador_de_lixo)
    )





# ---------------------------------------------------------------------------
# Pooled LLM for Mistral
# ---------------------------------------------------------------------------
from langchain_core.runnables import Runnable

class PooledLLM(Runnable):
    """A thread-safe proxy that distributes requests across a pool of LLMs."""
    _global_counters = {} # Shared counter per pool name
    _lock = threading.Lock()

    def __init__(self, llms, model_name="pool"):
        self.llms = llms
        self.model_name = model_name
        
        with PooledLLM._lock:
            if model_name not in PooledLLM._global_counters:
                PooledLLM._global_counters[model_name] = 0

    def get_next(self):
        with PooledLLM._lock:
            idx = PooledLLM._global_counters[self.model_name] % len(self.llms)
            PooledLLM._global_counters[self.model_name] += 1
            llm = self.llms[idx]
            
            # Log specific model being used for debugging
            model_id = getattr(llm, "model", getattr(llm, "model_name", "unknown"))
            log.debug("Pool '%s' selecionando modelo: %s", self.model_name, model_id)
            return llm

    def with_structured_output(self, schema, **kwargs):
        return PooledLLM(
            llms=[llm.with_structured_output(schema, **kwargs) for llm in self.llms],
            model_name=self.model_name
        )

    def invoke(self, input, config=None, **kwargs):
        return self.get_next().invoke(input, config=config, **kwargs)

    def batch(self, inputs, config=None, **kwargs):
        # LangChain's default batch uses threading. 
        # Our get_next() is thread-safe, so each item in the batch will pick a different model.
        return super().batch(inputs, config=config, **kwargs)


def init_llm(provider: str = "vllm", model_name: str = "Meta-Llama-3.1-8B-Instruct", max_retries: int = 5):
    """Returns (llm, model_name). Supports vllm, gemini, mistral, and deepseek without fallback."""
    provider = provider.lower()
    
    if provider == "gemini":
        try:
            paid_key = os.environ.get("GEMINI_API_KEY_PAID") or os.environ.get("GEMINI_API_KEY")
            free_key = os.environ.get("GEMINI_API_KEY_FREE")

            # PAID key usa modelo fixo configurável via env
            # FREE key descobre todos os modelos disponíveis dinamicamente
            paid_model = os.environ.get("GEMINI_PAID_MODEL", model_name)

            if free_key or paid_key:
                llm = GeminiPoolLLM.build(
                    free_key=free_key,
                    paid_key=paid_key,
                    paid_model=paid_model,
                    max_retries=max_retries,
                    max_output_tokens=32768,
                    probe=True,
                )
                effective_model = f"gemini-pool({len(llm._entries)} modelos)"
                log.info("[GeminiPool] Pronto: %s", effective_model)
            else:
                # Fallback legado: usa GEMINI_API_KEY genérica (sem pool)
                log.warning("Nenhuma GEMINI_API_KEY_FREE/PAID encontrada. Usando GEMINI_API_KEY genérica.")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.5,
                    max_retries=max_retries,
                    max_output_tokens=32768,
                )
                effective_model = model_name
                log.info("LLM iniciado: Gemini %s", model_name)

            return llm, effective_model
        except Exception as exc:
            log.error("Falha ao iniciar Gemini pool: %s", exc)
            raise

    elif provider == "mistral":
        if model_name == "mistral-pool":
            pool_models = [
                "mistral-large-latest",
                "mistral-medium-latest",
                "mistral-small-latest",
                "open-mistral-nemo",
                "ministral-8b-latest",
                "ministral-3b-latest",
                "codestral-latest",
            ]
            llms = []
            api_key = os.environ.get("MISTRAL_API_KEY", "MISTRAL_API_KEY_NOT_SET")
            for m in pool_models:
                # Mistral output limits vary, 8192 is the safe upper bound for newer models
                llms.append(ChatMistralAI(model=m, temperature=0.5, max_retries=max_retries, mistral_api_key=api_key, max_tokens=8192))
            log.info("Iniciando Mistral Pool com %d modelos (max_tokens=8192)", len(llms))

            return PooledLLM(llms, model_name="mistral-pool"), "mistral-pool"

        try:
            llm = ChatMistralAI(
                model=model_name, 
                temperature=0.5, 
                max_retries=max_retries,
                mistral_api_key=os.environ.get("MISTRAL_API_KEY", "MISTRAL_API_KEY_NOT_SET"),
                max_tokens=8192
            )
            log.info("LLM iniciado com sucesso: Mistral %s (max_tokens=8192)", model_name)
            return llm, model_name
        except Exception as exc:
            log.error("Falha ao iniciar Mistral %s: %s", model_name, exc)
            raise

    elif provider == "deepseek":
        try:
            # DeepSeek V3 supports 8000 output tokens
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.5,
                openai_api_base="https://api.deepseek.com",
                openai_api_key=os.environ.get("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_NOT_SET"),
                max_retries=max_retries,
                max_tokens=8192
            )
            log.info("LLM iniciado com sucesso: DeepSeek %s (max_tokens=8192)", model_name)
            return llm, model_name
        except Exception as exc:
            log.error("Falha ao iniciar DeepSeek %s: %s", model_name, exc)
            raise

    elif provider == "vllm":
        vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        actual_model = model_name
        if model_name in ("granite4.1:8b", "granite4.1:3b"):
            actual_model = "ibm-granite/granite-4.1-8b-fp8"
        elif model_name == "llama3.2:3b":
            actual_model = "meta-llama/Llama-3.2-3B-Instruct"

        try:
            # vLLM local can usually handle larger outputs if memory allows
            llm = ChatOpenAI(
                model=actual_model,
                temperature=0.5,
                openai_api_base=f"{vllm_url}/v1",
                openai_api_key="token-vllm-or-anything",
                max_retries=max_retries,
                max_tokens=16384
            )
            log.info("LLM iniciado com sucesso: vLLM %s (max_tokens=16384)", actual_model)
            return llm, actual_model
        except Exception as exc:
            log.error("Falha ao iniciar vLLM %s: %s", actual_model, exc)
            raise
            
    else:
        raise ValueError(f"Provider de LLM desconhecido: {provider}")


# ---------------------------------------------------------------------------
# Core processing function (used by both DAGs)
# ---------------------------------------------------------------------------
def process_pending_files(
    bucket_name: str = "mt-airflow",
    raw_prefix: str = "raw_corpus/",
    out_prefix: str = "datasets/pt-br_Q&A/",
    limit: int | None = None,
    provider: str = "vllm",
    model_name: str = "Meta-Llama-3.1-8B-Instruct",
    max_concurrency: int = 4,
    rpm: int | None = None,
    rps: float | None = None,
) -> dict:
    """
    Descobre quais parquets ainda não foram convertidos em JSONL e processa.

    Returns a summary dict pushed to XCom:
        {
            "files_found":      int,
            "files_skipped":    int,
            "files_processed":  int,
            "rows_total":       int,
            "rows_discarded":   int,
            "qa_generated":     int,
            "errors":           list[str],
        }
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # ---- listagem --------------------------------------------------------
    log.info("Listando parquets em gs://%s/%s …", bucket_name, raw_prefix)
    raw_blobs = [b.name for b in bucket.list_blobs(prefix=raw_prefix) if b.name.endswith(".parquet")]
    log.info("%d arquivo(s) parquet encontrado(s).", len(raw_blobs))

    log.info("Listando JSONLs já gerados em gs://%s/%s …", bucket_name, out_prefix)
    out_basenames = {
        os.path.basename(b.name)
        for b in bucket.list_blobs(prefix=out_prefix)
        if b.name.endswith(".jsonl")
    }

    pending = [rf for rf in raw_blobs if os.path.basename(rf).replace(".parquet", ".jsonl") not in out_basenames]
    skipped = len(raw_blobs) - len(pending)
    log.info("%d arquivo(s) já processado(s) (skip). %d pendente(s).", skipped, len(pending))

    if limit is not None and limit > 0:
        pending = pending[:limit]
        log.info("Limitando a %d arquivo(s) (limit=%d).", len(pending), limit)

    if not pending:
        log.info("Nada a processar. Encerrando.")
        return {
            "files_found": len(raw_blobs),
            "files_skipped": skipped,
            "files_processed": 0,
            "rows_total": 0,
            "rows_discarded": 0,
            "qa_generated": 0,
            "errors": [],
        }

    # ---- modelo ----------------------------------------------------------
    llm, model_name = init_llm(provider=provider, model_name=model_name)
    pipeline = build_pipeline(llm)

    # ---- contadores ------------------------------------------------------
    total_rows = 0
    total_discarded = 0
    total_qa = 0
    files_processed = 0
    errors = []

    # Configuração de taxa (RPM / RPS)
    delay_per_request = 0
    if rps and rps > 0:
        delay_per_request = 1.0 / rps
        log.info("Rate limiting ativo: %.2f RPS (delay: %.2fs por req)", rps, delay_per_request)
    elif rpm and rpm > 0:
        delay_per_request = 60.0 / rpm
        log.info("Rate limiting ativo: %d RPM (delay: %.2fs por req)", rpm, delay_per_request)

    for rf in pending:
        base_name = os.path.basename(rf)
        out_name = base_name.replace(".parquet", ".jsonl")
        local_parquet = f"/tmp/{base_name}"
        local_jsonl = f"/tmp/{out_name}"

        log.info("⬇️  Baixando %s …", rf)
        try:
            bucket.blob(rf).download_to_filename(local_parquet)
        except Exception as exc:
            msg = f"Erro ao baixar {rf}: {exc}"
            log.error(msg)
            errors.append(msg)
            continue

        df = pl.read_parquet(local_parquet)
        n_rows = len(df)
        total_rows += n_rows
        log.info("📄 %s — %d linhas.", base_name, n_rows)

        if "text" not in df.columns:
            msg = f"{base_name}: sem coluna 'text'. Pulando."
            log.warning(msg)
            errors.append(msg)
            os.remove(local_parquet)
            continue

        # Ingerir corpus brutas no YugabyteDB
        try:
            with get_db_connection() as db_conn:
                save_corpus_to_db(db_conn, df, rf)
        except Exception as db_exc:
            log.error("Erro ao salvar corpus no YugabyteDB: %s", db_exc)

        results_jsonl = []
        file_discarded = 0
        file_qa = 0

        rows_to_process = []
        for row_idx, row in enumerate(df.iter_rows(named=True)):
            texto = row.get("text", "")
            source = row.get("url", row.get("title", ""))
            if not texto:
                log.debug("Linha %d vazia, pulando.", row_idx)
                continue
            
            rows_to_process.append({
                "row_idx": row_idx,
                "texto": texto,
                "source": source
            })

        n_rows_to_process = len(rows_to_process)
        total_batches = (n_rows_to_process + max_concurrency - 1) // max_concurrency
        log.info("🚀 Iniciando processamento de %d linhas (%d lotes de %d) do arquivo %s.", n_rows_to_process, total_batches, max_concurrency, base_name)

        for batch_idx, i in enumerate(range(0, n_rows_to_process, max_concurrency)):
            batch = rows_to_process[i : i + max_concurrency]
            
            def execute_batch(items):
                batch_inputs = [{"texto": item["texto"]} for item in items]
                return pipeline.batch(
                    batch_inputs, 
                    config={"max_concurrency": max_concurrency},
                    return_exceptions=True
                )

            start_batch = time.time()
            log.info("📦 Processando lote %d/%d (%d itens) do arquivo %s...", batch_idx + 1, total_batches, len(batch), base_name)
            
            try:
                batch_results = execute_batch(batch)
            except Exception as exc:
                log.exception("❌ Erro fatal ao processar lote %d/%d do arquivo %s.", batch_idx + 1, total_batches, base_name)
                for item in batch:
                    errors.append(f"{base_name}[{item['row_idx']}]: Lote falhou — {exc}")
                continue

            # Identifica itens para reprocessar (erros transientes ou de schema)
            to_retry = []
            final_results = [None] * len(batch)
            
            for idx, res in enumerate(batch_results):
                if isinstance(res, Exception):
                    err_details = str(res)
                    # Se for erro de API ou Schema, tentamos mais uma vez
                    if "ValidationError" in err_details or "JSON" in err_details or "parsing" in err_details.lower() or "timeout" in err_details.lower() or "limit" in err_details.lower():
                        to_retry.append((idx, batch[idx]))
                    else:
                        final_results[idx] = res # Erro crítico, não tenta de novo
                else:
                    final_results[idx] = res

            if to_retry:
                log.info("🔄 Tentando reprocessar %d itens que falharam no lote %d...", len(to_retry), batch_idx + 1)
                retry_items = [item for _, item in to_retry]
                retry_results = execute_batch(retry_items)
                for (orig_idx, _), retry_res in zip(to_retry, retry_results):
                    final_results[orig_idx] = retry_res

            batch_qa_count = 0
            for item, res in zip(batch, final_results):
                row_idx = item["row_idx"]
                source = item["source"]
                texto = item["texto"]

                if isinstance(res, Exception):
                    # Identifica o tipo de erro final
                    err_type = "API_ERROR"
                    err_details = str(res)
                    
                    if "ValidationError" in err_details or "JSON" in err_details or "parsing" in err_details.lower():
                        err_type = "SCHEMA_MISMATCH"
                        log.error("❌ Linha %d: Modelo gerou dados fora do padrão (Schema Mismatch) após retry. Erro: %s", row_idx, err_details)
                    elif any(kw in err_details.lower() for kw in ["too large", "context_length", "context length", "maximum context length", "400"]):
                        # Erros de 400 da Mistral/OpenAI costumam ser tamanho de contexto
                        err_type = "CONTEXT_EXCEEDED"
                        log.error("❌ Linha %s: Chunk ainda excede o limite de contexto. Erro: %s", row_idx, err_details)
                    else:
                        log.error("❌ Linha %s: Falha na API/Rede após retry. Erro: %s", row_idx, err_details)
                    
                    msg = f"{base_name}[{row_idx}] [{err_type}]: {err_details}"
                    errors.append(msg)
                    continue

                if not isinstance(res, dict):
                    log.warning("⚠️ Linha %d — resultado inválido: %s", row_idx, type(res))
                    continue

                status = res.get("status_pipeline", "desconhecido")

                if status == "descartado_para_revisao":
                    file_discarded += 1
                    log.info("🗑️ Linha %d: Todos os samples foram descartados (Texto de baixa utilidade).", row_idx)
                elif status == "processado_com_sucesso":
                    instrucoes = res.get("dataset_instrucoes") or []
                    if not instrucoes:
                        log.warning("⚠️ Linha %d: Processado com sucesso mas nenhum Q&A foi gerado.", row_idx)
                        continue
                        
                    for qa in instrucoes:
                        resposta_obj = qa["response"]
                        results_jsonl.append({
                            "question":  qa["instruction"],
                            "reasoning": get_field(resposta_obj, "reasoning", ""),
                            "answer":    get_field(resposta_obj, "answer", ""),
                            "model":     model_name,
                            "source":    source,
                        })
                        file_qa += 1
                        batch_qa_count += 1
                    log.debug("✨ Linha %d → %d Q&As gerados.", row_idx, len(instrucoes or []))
                else:
                    log.warning("❓ Linha %d — status inesperado: %s", row_idx, status)

            elapsed_batch = time.time() - start_batch
            log.info(
                "⏱️ Lote %d/%d concluído em %.2fs. (%d Q&As gerados neste lote. Total do arquivo: %d)", 
                batch_idx + 1, total_batches, elapsed_batch, batch_qa_count, file_qa
            )

            # Respeita RPM / RPS
            if delay_per_request > 0:
                total_delay = delay_per_request * len(batch)
                if elapsed_batch < total_delay:
                    wait_time = total_delay - elapsed_batch
                    log.debug("⏳ Dormindo %.2fs para respeitar rate limit...", wait_time)
                    time.sleep(wait_time)

        total_discarded += file_discarded
        total_qa += file_qa
        log.info(
            "✅ %s processado: %d Q&As gerados, %d descartados.",
            base_name, file_qa, file_discarded,
        )

        # ---- salva JSONL -------------------------------------------------
        with open(local_jsonl, "w", encoding="utf-8") as f:
            for item in results_jsonl:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Ingerir Q&A no YugabyteDB
        try:
            with get_db_connection() as db_conn:
                save_qa_to_db(db_conn, results_jsonl, rf)
        except Exception as db_exc:
            log.error("Erro ao salvar Q&A no YugabyteDB: %s", db_exc)

        log.info("⬆️  Enviando %s para gs://%s/%s%s …", out_name, bucket_name, out_prefix, out_name)
        bucket.blob(f"{out_prefix}{out_name}").upload_from_filename(local_jsonl)
        log.info("☁️  Upload concluído.")
        files_processed += 1

        # cleanup
        for f in (local_parquet, local_jsonl):
            if os.path.exists(f):
                os.remove(f)

    summary = {
        "files_found":     len(raw_blobs),
        "files_skipped":   skipped,
        "files_processed": files_processed,
        "rows_total":      total_rows,
        "rows_discarded":  total_discarded,
        "qa_generated":    total_qa,
        "errors":          errors,
    }
    log.info("=== RESUMO FINAL === %s", json.dumps(summary, ensure_ascii=False))
    return summary


# ---------------------------------------------------------------------------
# CLI entrypoint (para uso sem Airflow)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CLI entrypoint (Daemon do Valkey para geração de Q&A)
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    redis_url = os.environ.get("REDIS_URL", "redis://valkey-primary.default.svc.cluster.local:6379")
    provider = os.environ.get("LLM_PROVIDER", "vllm")
    model_name = os.environ.get("LLM_MODEL_NAME", "Meta-Llama-3.1-8B-Instruct")
    
    log.info("Iniciando Daemon do QA Generator...")
    log.info("Provedor LLM: %s, Modelo: %s", provider, model_name)
    log.info("Conectando ao Valkey em: %s", redis_url)
    
    r = redis.Redis.from_url(redis_url)
    r.ping()
    log.info("Conexão com Valkey estabelecida com sucesso.")
    
    # Inicializar LLM e Pipeline
    llm, actual_model = init_llm(provider=provider, model_name=model_name)
    pipeline = build_pipeline(llm)
    log.info("Pipeline de Q&A montado com sucesso.")
    
    log.info("Aguardando tarefas na fila 'qa_queue'...")
    
    while True:
        try:
            # 1. Obter tarefa da fila
            result = r.brpop("qa_queue", timeout=5)
            if not result:
                continue
                
            _, payload_json = result
            payload = json.loads(payload_json)
            corpus_id = payload.get("id")
            raw_id = payload.get("raw_id")
            
            if not corpus_id:
                log.warning("Payload de QA inválido: %s", payload_json)
                continue
                
            log.info("Processando QA para o Corpus ID: %s (Raw ID: %s)", corpus_id, raw_id)
            
            # 2. Ler texto limpo do YugabyteDB
            db_conn = get_db_connection()
            row = None
            try:
                with db_conn.cursor() as cur:
                    cur.execute("SELECT text, url, title FROM corpus WHERE id = %s;", (corpus_id,))
                    row = cur.fetchone()
            finally:
                db_conn.close()
                
            if not row:
                log.warning("Corpus ID %s não encontrado na tabela 'corpus'. Pulando.", corpus_id)
                continue
                
            text, url, title = row
            source = url if url else (title if title else "unknown")
            
            # 3. Invocar pipeline de geração de Q&A
            t0 = time.time()
            res = pipeline.invoke({"texto": text})
            elapsed = time.time() - t0
            
            status = res.get("status_pipeline", "desconhecido")
            log.info("Processamento da LLM concluído em %.2fs. Status: %s", elapsed, status)
            
            # 4. Gravar resultados no YugabyteDB
            db_conn = get_db_connection()
            try:
                with db_conn.cursor() as cur:
                    if status == "processado_com_sucesso":
                        import uuid
                        from openai import OpenAI
                        
                        def generate_mistral_embeddings(texts: list[str]) -> list[list[float]]:
                            api_key = os.environ.get("MISTRAL_API_KEY", "")
                            base_url = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
                            model = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")
                            if not api_key:
                                log.warning("MISTRAL_API_KEY não definida. Retornando embeddings zerados.")
                                return [[0.0] * 1024 for _ in texts]
                            client = OpenAI(api_key=api_key, base_url=base_url)
                            try:
                                response = client.embeddings.create(model=model, input=texts)
                                return [e.embedding for e in response.data]
                            except Exception as e:
                                log.error("Erro ao gerar embeddings no Mistral: %s", e)
                                raise
                        
                        instrucoes = res.get("dataset_instrucoes") or []
                        qa_items = []
                        
                        meta = {
                            "corpus_id": corpus_id,
                            "raw_id": raw_id
                        }
                        
                        for qa in list(instrucoes):
                            question = clean_string(qa.get("instruction"))
                            resp_obj = qa.get("response") or {}
                            reasoning = clean_string(resp_obj.get("reasoning", ""))
                            answer = clean_string(resp_obj.get("answer", ""))
                            
                            if question and answer:
                                qa_items.append({
                                    "id": str(uuid.uuid4()),
                                    "question": question,
                                    "reasoning": reasoning,
                                    "answer": answer
                                })
                                
                        if qa_items:
                            questions = [item["question"] for item in qa_items]
                            answers = [item["answer"] for item in qa_items]
                            
                            try:
                                q_embeddings = generate_mistral_embeddings(questions)
                                a_embeddings = generate_mistral_embeddings(answers)
                            except Exception as emb_err:
                                log.error("Falha ao obter embeddings de Mistral para Q&A: %s", emb_err)
                                raise emb_err
                            
                            rows_to_insert = []
                            chunks_to_insert = []
                            for item, q_emb, a_emb in zip(qa_items, q_embeddings, a_embeddings):
                                rows_to_insert.append((
                                    item["id"],
                                    item["question"],
                                    item["reasoning"],
                                    item["answer"],
                                    actual_model,
                                    source,
                                    q_emb,
                                    json.dumps(meta)
                                ))
                                chunks_to_insert.append((
                                    item["id"],
                                    'qa_dataset',
                                    0,
                                    item["answer"],
                                    a_emb
                                ))
                                
                            from psycopg2.extras import execute_values
                            execute_values(
                                cur,
                                """
                                INSERT INTO qa_dataset (id, question, reasoning, answer, model, source, embedding, metadata)
                                VALUES %s
                                """,
                                rows_to_insert
                            )
                            execute_values(
                                cur,
                                """
                                INSERT INTO public.embeddings_chunks (record_id, table_name, chunk_index, chunk_text, embedding)
                                VALUES %s
                                """,
                                chunks_to_insert
                            )
                            log.info("✅ Inseridos %d Q&As na tabela qa_dataset e em embeddings_chunks.", len(rows_to_insert))
                            
                    # Marcar como processado na tabela raw_corpus
                    if raw_id:
                        cur.execute("UPDATE raw_corpus SET processed_qa = TRUE WHERE id = %s;", (raw_id,))
                        
                db_conn.commit()
            except Exception as dbe:
                log.error("Erro ao gravar Q&As no banco para o Corpus ID %s: %s", corpus_id, dbe)
                db_conn.rollback()
            finally:
                db_conn.close()
                
        except redis.exceptions.ConnectionError:
            log.warning("Conexão com Valkey perdida no QA Daemon. Tentando reconectar...")
            time.sleep(2)
        except Exception as e:
            log.error("Erro no loop principal do QA Daemon: %s", e)
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    import redis  # Garantir import local
    main()

