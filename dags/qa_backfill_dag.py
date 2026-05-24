from __future__ import annotations

import logging
import sys
import os
from datetime import datetime, timezone

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.models.param import Param

log = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("OUTPUT_BUCKET", "mt-airflow")
RAW_PREFIX = "raw_corpus/"
OUT_PREFIX = "datasets/pt-br_Q&A/"

SCRIPTS_PATH = "/opt/airflow/dags/repo/dags/scripts"
DAGS_PATH = "/opt/airflow/dags/repo/dags"


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

    from generate_qa import process_pending_files  # noqa: PLC0415

    params = context.get("params", {})
    llm_provider = params.get("llm_provider", "ollama")
    llm_model_selected = params.get("llm_model", "granite4.1:3b")
    custom_llm_model = params.get("custom_llm_model", "").strip()

    if llm_model_selected == "Customizado (digitar no campo abaixo)":
        if custom_llm_model:
            llm_model = custom_llm_model
        else:
            llm_model = "gemini-2.5-flash" if llm_provider == "gemini" else "granite4.1:3b"
    else:
        llm_model = llm_model_selected

    log.info("Iniciando backfill em gs://%s/%s com o provedor %s e modelo %s", GCS_BUCKET, RAW_PREFIX, llm_provider, llm_model)
    summary = process_pending_files(
        bucket_name=GCS_BUCKET,
        raw_prefix=RAW_PREFIX,
        out_prefix=OUT_PREFIX,
        provider=llm_provider,
        model_name=llm_model,
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
        "para todos os chunks que ainda não foram processados."
    ),
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,       # apenas acionamento manual
    catchup=False,
    max_active_runs=1,   # garante que não rode em paralelo
    default_args={"owner": "dataset-builder"},
    params={
        "llm_provider": Param("ollama", type="string", enum=["ollama", "gemini"], description="Provedor de LLM a ser utilizado"),
        "llm_model": Param(
            "granite4.1:3b",
            type="string",
            enum=[
                "granite4.1:3b",
                "granite4.1:8b",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-2.0-flash-lite",
                "flash-latest",
                "flash-lite-latest",
                "pro-latest",
                "Customizado (digitar no campo abaixo)",
            ],
            description="Modelo da LLM a ser utilizado (ou escolha 'Customizado' para digitar abaixo)",
        ),
        "custom_llm_model": Param(
            "",
            type="string",
            description="Caso tenha escolhido 'Customizado' no campo acima, digite o modelo (Ex: gemini-3.5-flash, gemini-3.1-flash-lite)",
        ),
    },
    tags=["dataset-builder", "qa", "backfill"],
) as dag:

    backfill_qa = PythonOperator(
        task_id="backfill_qa",
        python_callable=_backfill,
    )

    backfill_qa
