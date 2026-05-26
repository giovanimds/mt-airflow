from __future__ import annotations

import logging
import sys
import os
from datetime import datetime, timezone

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectsWithPrefixExistenceSensor
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

def _process_qa(**context):
    """
    Verifica quais chunks ainda não foram processados e gera Q&A apenas para eles.
    Pusha um dict de resumo via XCom com as chaves:
        files_found, files_skipped, files_processed,
        rows_total, rows_discarded, qa_generated, errors
    """
    for p in (SCRIPTS_PATH, DAGS_PATH):
        if p not in sys.path:
            sys.path.insert(0, p)

    from generate_qa import process_pending_files  # noqa: PLC0415

    params = context.get("params", {})
    llm_model_selected = params.get("llm_model", "vllm: Meta-Llama-3.1-8B-Instruct")
    custom_llm_model = (params.get("custom_llm_model") or "").strip()
    limit = params.get("limit")
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
        "qa_generator_dag iniciado — verificando novos chunks em gs://%s/%s com o provedor %s e modelo %s (concurrency=%s, rpm=%s, rps=%s)",
        GCS_BUCKET, RAW_PREFIX, llm_provider, llm_model, max_concurrency, rpm, rps
    )

    summary = process_pending_files(
        bucket_name=GCS_BUCKET,
        raw_prefix=RAW_PREFIX,
        out_prefix=OUT_PREFIX,
        provider=llm_provider,
        model_name=llm_model,
        limit=limit,
        max_concurrency=max_concurrency,
        rpm=rpm,
        rps=rps,
    )

    ti = context["ti"]
    ti.xcom_push(key="summary", value=summary)

    log.info(
        "Ciclo concluído — processados: %d / %d  |  Q&As: %d  |  erros: %d",
        summary["files_processed"],
        summary["files_found"],
        summary["qa_generated"],
        len(summary["errors"]),
    )

    if summary["errors"]:
        log.warning("Erros neste ciclo:\n%s", "\n".join(summary["errors"]))

    return summary


with DAG(
    dag_id="qa_generator_dag",
    description=(
        "Monitora o bucket GCS e gera Q&A para cada novo chunk de corpus. "
        "Roda a cada 30 min; processa apenas arquivos ainda não convertidos."
    ),
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule="*/30 * * * *",
    catchup=False,
    max_active_runs=1,
    params={
        "limit": Param(2, type=["integer", "null"], description="Limite de arquivos a processar nesta rodada. Deixe nulo ou 0 para sem limite."),
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
    tags=["dataset-builder", "qa", "generation", "sensor"],
) as dag:

    # Sensor: confirma que há ao menos 1 parquet no prefixo.
    # mode=reschedule libera o worker slot enquanto aguarda.
    aguardar_novos_chunks = GCSObjectsWithPrefixExistenceSensor(
        task_id="aguardar_novos_chunks",
        bucket=GCS_BUCKET,
        prefix=RAW_PREFIX,
        google_cloud_conn_id="google_cloud_default",
        mode="reschedule",
        poke_interval=120,
        timeout=25 * 60,
        soft_fail=True,   # não falha o DAG se o bucket estiver vazio
    )

    processar_novos_chunks = PythonOperator(
        task_id="processar_novos_chunks",
        python_callable=_process_qa,
    )

    aguardar_novos_chunks >> processar_novos_chunks
