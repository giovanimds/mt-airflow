from __future__ import annotations

import logging
import sys
import os
from datetime import datetime, timezone

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import Param

log = logging.getLogger(__name__)

SCRIPTS_PATH = "/opt/airflow/dags/repo/dags/scripts"
DAGS_PATH = "/opt/airflow/dags/repo/dags"

# Garante que scripts/ esteja no sys.path para o DAG parser
_local_scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if _local_scripts_path not in sys.path:
    sys.path.insert(0, _local_scripts_path)

def _run_backfill(**context):
    for p in (SCRIPTS_PATH, DAGS_PATH):
        if p not in sys.path:
            sys.path.insert(0, p)

    from bucket_to_db_backfill import run_backfill  # noqa: PLC0415

    params = context.get("params", {})
    limit_files = params.get("limit_files", 0)
    push_to_valkey = params.get("push_to_valkey", True)

    limit = limit_files if limit_files > 0 else None

    log.info(f"Disparando backfill manual do bucket: limit_files={limit}, push_to_valkey={push_to_valkey}")
    
    files, inserted = run_backfill(limit_files=limit, push_to_valkey=push_to_valkey)
    
    log.info(f"Backfill concluído: {files} arquivos processados, {inserted} novos registros salvos no YugabyteDB.")
    return {"files_processed": files, "inserted_count": inserted}

with DAG(
    dag_id="bucket_to_db_backfill_dag",
    description="Lê dados brutos (Parquet) do GCS Bucket, salva no YugabyteDB e enfileira no Valkey.",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,  # Apenas acionamento manual
    catchup=False,
    max_active_runs=1,
    params={
        "limit_files": Param(
            0,
            type="integer",
            description="Limite de arquivos Parquet a processar (use 0 para processar todos).",
        ),
        "push_to_valkey": Param(
            True,
            type="boolean",
            description="Se marcado, enfileira os novos registros na fila Valkey raw_corpus_queue para processamento imediato.",
        ),
    },
    tags=["dataset-builder", "backfill", "gcs", "yugabyte"],
) as dag:

    run_bf = PythonOperator(
        task_id="run_bucket_backfill",
        python_callable=_run_backfill,
    )

    run_bf
