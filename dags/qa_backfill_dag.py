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
    ollama_model = params.get("ollama_model", "granite4.1:3b")

    log.info("Iniciando backfill em gs://%s/%s com o modelo %s", GCS_BUCKET, RAW_PREFIX, ollama_model)
    summary = process_pending_files(
        bucket_name=GCS_BUCKET,
        raw_prefix=RAW_PREFIX,
        out_prefix=OUT_PREFIX,
        model_name=ollama_model,
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
        "ollama_model": Param("granite4.1:3b", type="string", description="Modelo Ollama a ser utilizado"),
    },
    tags=["dataset-builder", "qa", "backfill"],
) as dag:

    backfill_qa = PythonOperator(
        task_id="backfill_qa",
        python_callable=_backfill,
    )

    backfill_qa
