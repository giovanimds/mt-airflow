import os
import json
import logging
import traceback
import polars as pl
import urllib.request
import time

from google.cloud import storage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

log = logging.getLogger(__name__)

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

solucionador_prompt = ChatPromptTemplate.from_template(
    """Você é um pesquisador sênior respondendo a uma dúvida acadêmica.

    Escreva sua linha de raciocínio passo a passo (Identificação da intenção -> Raciocínio -> Resposta -> Revisão)
    e depois forneça a conclusão/resposta final direta e detalhada.

    Pergunta: {pergunta}
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

class QAResult(BaseModel):
    reasoning: str = Field(description="Sua linha de raciocínio passo a passo detalhada e profunda")
    answer: str = Field(description="Sua conclusão ou resposta final direta e detalhada")


def get_field(obj, field_name, default=""):
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline(llm):
    chain_perguntas = qa_prompt | llm.with_structured_output(QuestionsList)
    chain_resposta = solucionador_prompt | llm.with_structured_output(QAResult)

    def gerar_dataset_desacoplado(inputs):
        resultado_perguntas = chain_perguntas.invoke(inputs)
        lista_perguntas = get_field(resultado_perguntas, "perguntas", []) if resultado_perguntas else []

        dataset = []
        for p in lista_perguntas:
            p_text = get_field(p, "pergunta", "")
            if not p_text:
                continue
            resposta = chain_resposta.invoke({"pergunta": p_text})
            if resposta:
                dataset.append({"instruction": p_text, "response": resposta})
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
        return RunnableParallel(**tarefas_paralelas)

    return (
        abstract_prompt | llm.with_structured_output(AbstractResult)
        | {
            "fase_1": lambda x: x,
            "fase_2": {"abstract": lambda x: get_field(x, "abstract", "") if x else ""} | meta_prompt | llm.with_structured_output(MetaResult),
        }
        | RunnableLambda(roteador_de_lixo)
    )





def init_llm(provider: str = "vllm", model_name: str = "Meta-Llama-3.1-8B-Instruct", max_retries: int = 5):
    """Returns (llm, model_name). Supports vllm, gemini, mistral, and deepseek without fallback."""
    provider = provider.lower()
    
    if provider == "gemini":
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7, max_retries=max_retries)
            log.info("LLM iniciado com sucesso: Gemini %s (max_retries=%d)", model_name, max_retries)
            return llm, model_name
        except Exception as exc:
            log.error("Falha ao iniciar Gemini %s: %s", model_name, exc)
            raise

    elif provider == "mistral":
        try:
            llm = ChatMistralAI(
                model=model_name, 
                temperature=0.7, 
                max_retries=max_retries,
                mistral_api_key=os.environ.get("MISTRAL_API_KEY", "MISTRAL_API_KEY_NOT_SET")
            )
            log.info("LLM iniciado com sucesso: Mistral %s (max_retries=%d)", model_name, max_retries)
            return llm, model_name
        except Exception as exc:
            log.error("Falha ao iniciar Mistral %s: %s", model_name, exc)
            raise

    elif provider == "deepseek":
        try:
            # DeepSeek uses an OpenAI-compatible API
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                openai_api_base="https://api.deepseek.com",
                openai_api_key=os.environ.get("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_NOT_SET"),
                max_retries=max_retries
            )
            log.info("LLM iniciado com sucesso: DeepSeek %s (max_retries=%d)", model_name, max_retries)
            return llm, model_name
        except Exception as exc:
            log.error("Falha ao iniciar DeepSeek %s: %s", model_name, exc)
            raise

    elif provider == "vllm":
        vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        actual_model = model_name
        # Keep custom mapping if needed, though dynamic models should provide correct names
        if model_name in ("granite4.1:8b", "granite4.1:3b"):
            actual_model = "ibm-granite/granite-4.1-8b-fp8"
        elif model_name == "llama3.2:3b":
            actual_model = "meta-llama/Llama-3.2-3B-Instruct"

        try:
            log.info("Testando conexao com vLLM em: %s", vllm_url)
            # Use urllib to test without external dependency or just init
            health_url = f"{vllm_url}/health"
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status != 200:
                    raise Exception(f"Status HTTP {response.status}")
            
            llm = ChatOpenAI(
                model=actual_model,
                temperature=0.7,
                openai_api_base=f"{vllm_url}/v1",
                openai_api_key="token-vllm-or-anything",
                max_retries=max_retries
            )
            log.info("LLM iniciado com sucesso: vLLM %s em %s (max_retries=%d)", actual_model, vllm_url, max_retries)
            return llm, actual_model
        except Exception as exc:
            log.error("Falha ao iniciar vLLM em %s com modelo %s: %s", vllm_url, actual_model, exc)
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
            batch_inputs = [{"texto": item["texto"]} for item in batch]

            start_batch = time.time()
            log.info("📦 Processando lote %d/%d (%d itens) do arquivo %s...", batch_idx + 1, total_batches, len(batch), base_name)
            
            try:
                # O LangChain usará max_concurrency internamente se o runnable suportar,
                # mas como estamos fatiando manualmente aqui, o batch do runnable
                # processará todos os itens do nosso 'batch' em paralelo (se configurado).
                batch_results = pipeline.batch(
                    batch_inputs, 
                    config={"max_concurrency": max_concurrency},
                    return_exceptions=True
                )
            except Exception as exc:
                log.exception("❌ Erro fatal ao processar lote %d/%d do arquivo %s.", batch_idx + 1, total_batches, base_name)
                for item in batch:
                    errors.append(f"{base_name}[{item['row_idx']}]: Lote falhou — {exc}")
                continue

            batch_qa_count = 0
            for item, res in zip(batch, batch_results):
                row_idx = item["row_idx"]
                source = item["source"]
                texto = item["texto"]

                if isinstance(res, Exception):
                    log.exception("⚠️ Erro na linha %d do arquivo %s.", row_idx, base_name)
                    msg = f"{base_name}[{row_idx}]: {res}"
                    errors.append(msg)
                    continue

                if not isinstance(res, dict):
                    log.warning("⚠️ Linha %d — resultado inválido: %s", row_idx, type(res))
                    continue

                status = res.get("status_pipeline", "desconhecido")

                if status == "descartado_para_revisao":
                    file_discarded += 1
                    log.debug("🗑️ Linha %d descartada (baixa utilidade).", row_idx)
                elif status == "processado_com_sucesso":
                    instrucoes = res.get("dataset_instrucoes") or []
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
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    bucket_name = os.environ.get("OUTPUT_BUCKET", "mt-airflow")
    summary = process_pending_files(bucket_name=bucket_name)
    log.info("Concluído. Resumo: %s", summary)


if __name__ == "__main__":
    main()
