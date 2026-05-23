from __future__ import annotations

import logging
import sys
import os
from datetime import datetime, timezone

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectsWithPrefixExistenceSensor

log = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("OUTPUT_BUCKET", "mt-airflow")
RAW_PREFIX = "raw_corpus/"
OUT_PREFIX = "datasets/pt-br_Q&A/"

SCRIPTS_PATH = "/opt/airflow/dags/repo/dags/scripts"
DAGS_PATH = "/opt/airflow/dags/repo/dags"


def _process_new_chunks(**context):
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

    log.info(
        "qa_generator_dag iniciado — verificando novos chunks em gs://%s/%s",
        GCS_BUCKET, RAW_PREFIX,
    )

    summary = process_pending_files(
        bucket_name=GCS_BUCKET,
        raw_prefix=RAW_PREFIX,
        out_prefix=OUT_PREFIX,
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
    default_args={"owner": "dataset-builder"},
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
        python_callable=_process_new_chunks,
    )

    aguardar_novos_chunks >> processar_novos_chunks
