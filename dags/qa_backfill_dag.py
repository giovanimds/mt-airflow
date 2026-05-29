from __future__ import annotations

import logging
import sys
import os
from datetime import datetime, timezone

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import Param

log = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("OUTPUT_BUCKET", "mt-airflow")
RAW_PREFIX = "raw_corpus/"
OUT_PREFIX = "datasets/pt-br_Q&A/"

SCRIPTS_PATH = "/opt/airflow/dags/repo/dags/scripts"
DAGS_PATH = "/opt/airflow/dags/repo/dags"

# Garante que scripts/ esteja no sys.path para o DAG parser
_local_scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if _local_scripts_path not in sys.path:
    sys.path.insert(0, _local_scripts_path)

try:
    from model_utils import get_available_models
    AVAILABLE_MODELS = get_available_models()
except Exception as e:
    log.warning("Failed to load models dynamically: %s", e)
    AVAILABLE_MODELS = [
        "vllm: Meta-Llama-3.1-8B-Instruct",
        "gemini: gemini-2.0-flash",
        "mistral: mistral-large-latest",
        "deepseek: deepseek-chat",
        "Customizado (digitar no campo abaixo)"
    ]

# Versão do Script Principal: 2026-05-29 00:03:00
# (Atualize o comentário acima para forçar o Airflow a recarregar a DAG)

def _backfill(**context):
    """
    Processa TODOS os chunks parquet ainda sem JSONL correspondente.
    Pusha um dict de resumo via XCom com as chaves:
        files_found, files_skipped, files_processed,
        rows_total, rows_discarded, qa_generated, errors
    """
    # Garante que o módulo generate_qa seja encontrado
    for p in (SCRIPTS_PATH, DAGS_PATH):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Log logic to show script version
    try:
        script_path = os.path.join(SCRIPTS_PATH, "generate_qa.py")
        if os.path.exists(script_path):
            mtime = os.path.getmtime(script_path)
            dt_mtime = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            log.info("🚀 [SCRIPT VERSION] generate_qa.py atualizado em: %s", dt_mtime)
    except Exception:
        pass

    from generate_qa import process_pending_files  # noqa: PLC0415

    params = context.get("params", {})
    llm_model_selected = params.get("llm_model", "vllm: Meta-Llama-3.1-8B-Instruct")
    custom_llm_model = (params.get("custom_llm_model") or "").strip()
    max_concurrency = params.get("max_concurrency", 4)
    rpm = params.get("rpm")
    rps = params.get("rps")

    if llm_model_selected == "Customizado (digitar no campo abaixo)":
        llm_input = custom_llm_model
    else:
        llm_input = llm_model_selected

    # Parse provider and model (format: "provider: model_name")
    if ": " in llm_input:
        llm_provider, llm_model = llm_input.split(": ", 1)
    else:
        # Fallback if format is invalid
        llm_provider = "vllm"
        llm_model = llm_input or "Meta-Llama-3.1-8B-Instruct"

    log.info(
        "Iniciando backfill em gs://%s/%s com o provedor %s e modelo %s (concurrency=%s, rpm=%s, rps=%s)", 
        GCS_BUCKET, RAW_PREFIX, llm_provider, llm_model, max_concurrency, rpm, rps
    )
    summary = process_pending_files(
        bucket_name=GCS_BUCKET,
        raw_prefix=RAW_PREFIX,
        out_prefix=OUT_PREFIX,
        provider=llm_provider,
        model_name=llm_model,
        max_concurrency=max_concurrency,
        rpm=rpm,
        rps=rps,
    )

    # Pusha o resumo para XCom (visível na UI do Airflow)
    ti = context["ti"]
    ti.xcom_push(key="summary", value=summary)

    log.info(
        "Backfill concluído — processados: %d / %d  |  Q&As: %d  |  erros: %d",
        summary["files_processed"],
        summary["files_found"],
        summary["qa_generated"],
        len(summary["errors"]),
    )

    if summary["errors"]:
        log.warning("Erros encontrados:\n%s", "\n".join(summary["errors"]))

    return summary


with DAG(
    dag_id="qa_backfill_dag",
    description=(
        "Backfill manual: percorre todo o raw_corpus/ do bucket e gera Q&A "
        "para todos os chunks que ainda não foram processados. "
        "Dica: utilize 'mistral: mistral-pool' para maximizar a taxa de requisições."
    ),
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,       # apenas acionamento manual
    catchup=False,
    max_active_runs=1,   # garante que não rode em paralelo
    default_args={"owner": "dataset-builder"},
    params={
        "max_concurrency": Param(4, type="integer", minimum=1, description="Máximo de requisições paralelas (concorrência)"),
        "rpm": Param(None, type=["integer", "null"], description="Limite de Requisições por Minuto (RPM). Deixe vazio para sem limite."),
        "rps": Param(None, type=["number", "null"], description="Limite de Requisições por Segundo (RPS). Deixe vazio para sem limite."),
        "llm_model": Param(
            AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "vllm: Meta-Llama-3.1-8B-Instruct",
            type="string",
            enum=AVAILABLE_MODELS,
            description="Modelo da LLM a ser utilizado (ou escolha 'Customizado' para digitar abaixo)",
        ),
        "custom_llm_model": Param(
            None,
            type=["string", "null"],
            description="Caso tenha escolhido 'Customizado' no campo acima, digite no formato 'provedor: modelo' (Ex: gemini: gemini-1.5-flash)",
        ),
    },
    tags=["dataset-builder", "qa", "backfill"],
) as dag:

    backfill_qa = PythonOperator(
        task_id="backfill_qa",
        python_callable=_backfill,
    )

    backfill_qa
