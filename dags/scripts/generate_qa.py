import os
import json
import logging
import traceback
import polars as pl
import urllib.request

from google.cloud import storage
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
abstract_prompt = ChatPromptTemplate.from_template(
    """Você é um assistente especializado em mineração de dados acadêmicos. 
Analise o texto bruto fornecido e retorne APENAS um objeto JSON contendo duas chaves:
1. "abstract": O resumo do artigo científico isolado de qualquer texto de interface ou rodapé.
2. "keywords": Uma lista com as palavras-chave explícitas no texto.

Ignore completamente endereços, e-mails, instruções de navegação de sites e direitos de acesso.
Não traduza o texto, apenas extraia as informações solicitadas.

Texto Bruto:
{texto}
"""
)

meta_prompt = ChatPromptTemplate.from_template(
    """Com base no resumo acadêmico fornecido, extraia os metadados metodológicos no formato JSON estrito abaixo:

{{
  "publico_alvo": "",
  "conclusoes_principais": "",
  "eh_ponderativo": true/false,
  "eh_exploratorio": true/false,
  "eh_especulativo": true/false,
  "eh_embasado_em_dados": true/false,
  "eh_destilavel_pergunta_resposta": true/false,
  "eh_destilavel_chain_of_thought": true/false,
  "eh_desprovido_de_utilidade_pratica": true/false
}}

Regras:
- Se uma informação não estiver explícita, preencha com null ou array vazio.
- Responda estritamente com o JSON, sem explicações adicionais.
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

    Retorne APENAS uma lista JSON com o formato:
    [
      {{
        "pergunta": "Pergunta contextualizada em pt-BR"
      }}
    ]

    Resumo:
    {abstract}
    """
)

solucionador_prompt = ChatPromptTemplate.from_template(
    """Você é um pesquisador sênior respondendo a uma dúvida acadêmica.
    Explique sua linha de raciocínio passo a passo antes de dar a conclusão final.

    REGRAS:
    - Raciocinio: **Identificação da intenção** → **Raciocínando** → **Resposta** → **Revisão**
    - Conclusão final: A resposta final baseada no raciocínio

    Pergunta: {pergunta}
    """
)

json_parser = JsonOutputParser()
str_parser = StrOutputParser()


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline(llm):
    # Detecta se é ChatOllama e aplica o bind adequado para forçar JSON estruturado
    try:
        from langchain_ollama import ChatOllama
        is_ollama = isinstance(llm, ChatOllama)
    except ImportError:
        is_ollama = False

    if is_ollama:
        llm_json = llm.bind(format="json")
    else:
        try:
            llm_json = llm.bind(response_format={"type": "json_object"})
        except Exception:
            llm_json = llm

    chain_perguntas = qa_prompt | llm_json | json_parser
    chain_resposta = solucionador_prompt | llm | str_parser

    def gerar_dataset_desacoplado(inputs):
        resultado_perguntas = chain_perguntas.invoke(inputs)
        if isinstance(resultado_perguntas, dict):
            lista_perguntas = resultado_perguntas.get("perguntas", [])
        elif isinstance(resultado_perguntas, list):
            lista_perguntas = resultado_perguntas
        else:
            lista_perguntas = []

        dataset = []
        for p in lista_perguntas:
            p_text = p.get("pergunta", "") if isinstance(p, dict) else str(p)
            if not p_text:
                continue
            resposta = chain_resposta.invoke({"pergunta": p_text})
            dataset.append({"instruction": p_text, "response": resposta})
        return dataset

    def roteador_de_lixo(inputs):
        fase_1 = inputs.get("fase_1", {})
        fase_2 = inputs.get("fase_2", {})
        if not isinstance(fase_1, dict):
            fase_1 = {"abstract": str(fase_1) if fase_1 is not None else "", "keywords": []}
        if not isinstance(fase_2, dict):
            fase_2 = {}

        eh_lixo = fase_2.get("eh_desprovido_de_utilidade_pratica", False) or not fase_1.get("abstract")

        if eh_lixo:
            return {
                "fase_1": fase_1,
                "fase_2": fase_2,
                "dataset_instrucoes": None,
                "status_pipeline": "descartado_para_revisao",
            }

        destilacao_input = {"abstract": fase_1.get("abstract", "") if isinstance(fase_1, dict) else "", "quantidade": 3}
        tarefas_paralelas = {
            "fase_1": lambda _: fase_1,
            "fase_2": lambda _: fase_2,
        }

        # Geramos as Q&As para qualquer artigo com utilidade prática (não-lixo)
        tarefas_paralelas["dataset_instrucoes"] = (
            (lambda _: destilacao_input) | RunnableLambda(gerar_dataset_desacoplado)
        )

        tarefas_paralelas["status_pipeline"] = lambda _: "processado_com_sucesso"
        return RunnableParallel(**tarefas_paralelas)

    return (
        abstract_prompt | llm_json | json_parser
        | {
            "fase_1": lambda x: x,
            "fase_2": {"abstract": lambda x: x.get("abstract", "") if isinstance(x, dict) else ""} | meta_prompt | llm_json | json_parser,
        }
        | RunnableLambda(roteador_de_lixo)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_reasoning_and_answer(response_text: str):
    reasoning = response_text
    answer = ""
    if "Conclusão final" in response_text:
        parts = response_text.split("Conclusão final", 1)
        reasoning = parts[0].strip()
        answer = parts[1].strip()
        if answer.startswith(":"):
            answer = answer[1:].strip()
    return reasoning, answer


def init_llm(provider: str = "ollama", model_name: str = "granite4.1:3b"):
    """Returns (llm, model_name). Supports ollama and gemini, with fallback to Mistral."""
    if provider.lower() == "gemini":
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7)
            log.info("LLM iniciado com sucesso: Gemini %s", model_name)
            return llm, model_name
        except Exception as exc:
            log.warning("Gemini indisponivel %s (%s), usando Mistral como fallback.", model_name, exc)
            fallback_model_name = "ministral-3b-2512"
            llm = ChatMistralAI(model=fallback_model_name, temperature=0.7)
            log.info("LLM iniciado com sucesso: Mistral %s", fallback_model_name)
            return llm, fallback_model_name

    # Default: ollama
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        log.info("Testando conexao com Ollama em: %s", ollama_url)
        # Envia um GET rapido para testar se o Ollama esta online
        with urllib.request.urlopen(ollama_url, timeout=2) as response:
            if response.status == 200:
                llm = ChatOllama(model=model_name, temperature=0.7, base_url=ollama_url)
                log.info("LLM iniciado com sucesso: Ollama %s em %s", model_name, ollama_url)
                return llm, model_name
            else:
                raise Exception(f"Status HTTP {response.status}")
    except Exception as exc:
        log.warning("Ollama indisponivel em %s (%s), usando Mistral como fallback.", ollama_url, exc)
        fallback_model_name = "ministral-3b-2512"
        llm = ChatMistralAI(model=fallback_model_name, temperature=0.7)
        log.info("LLM iniciado com sucesso: Mistral %s", fallback_model_name)
        return llm, fallback_model_name


# ---------------------------------------------------------------------------
# Core processing function (used by both DAGs)
# ---------------------------------------------------------------------------
def process_pending_files(
    bucket_name: str = "mt-airflow",
    raw_prefix: str = "raw_corpus/",
    out_prefix: str = "datasets/pt-br_Q&A/",
    limit: int | None = None,
    provider: str = "ollama",
    model_name: str = "granite4.1:3b",
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

    if limit is not None:
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

        BATCH_SIZE = 4
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

        for i in range(0, len(rows_to_process), BATCH_SIZE):
            batch = rows_to_process[i : i + BATCH_SIZE]
            batch_inputs = [{"texto": item["texto"]} for item in batch]

            try:
                batch_results = pipeline.batch(batch_inputs, return_exceptions=True)
            except Exception as exc:
                log.exception("Erro fatal ao processar lote %d-%d do arquivo %s.", i, i + len(batch) - 1, base_name)
                for item in batch:
                    errors.append(f"{base_name}[{item['row_idx']}]: Lote falhou — {exc}")
                continue

            for item, res in zip(batch, batch_results):
                row_idx = item["row_idx"]
                source = item["source"]
                texto = item["texto"]

                if isinstance(res, Exception):
                    log.exception("Erro ao processar linha %d do arquivo %s. Texto original (primeiros 300 caracteres): %r", row_idx, base_name, texto[:300], exc_info=res)
                    msg = f"{base_name}[{row_idx}]: {res}"
                    errors.append(msg)
                    continue

                if not isinstance(res, dict):
                    log.warning("Linha %d — resultado inválido (não é dict): %s", row_idx, type(res))
                    continue

                status = res.get("status_pipeline", "desconhecido")

                if status == "descartado_para_revisao":
                    file_discarded += 1
                    log.debug("Linha %d descartada (baixa utilidade).", row_idx)
                elif status == "processado_com_sucesso":
                    instrucoes = res.get("dataset_instrucoes") or []
                    for qa in instrucoes:
                        reasoning, answer = parse_reasoning_and_answer(qa["response"])
                        results_jsonl.append({
                            "question":  qa["instruction"],
                            "reasoning": reasoning,
                            "answer":    answer,
                            "model":     model_name,
                            "source":    source,
                        })
                        file_qa += 1
                    log.debug("Linha %d → %d Q&As gerados.", row_idx, len(instrucoes or []))
                else:
                    log.warning("Linha %d — status inesperado: %s", row_idx, status)

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
