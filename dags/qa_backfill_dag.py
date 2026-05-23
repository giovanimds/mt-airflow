from __future__ import annotations

from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="qa_backfill_dag",
    description=(
        "Backfill manual: percorre todo o raw_corpus/ do bucket e gera Q&A "
        "para todos os chunks que ainda não foram processados."
    ),
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,           # apenas acionamento manual
    catchup=False,
    max_active_runs=1,       # garante que não rode em paralelo
    default_args={"owner": "dataset-builder"},
    tags=["dataset-builder", "qa", "backfill"],
) as dag:

    backfill_qa = BashOperator(
        task_id="backfill_qa",
        bash_command="python /opt/airflow/dags/repo/dags/scripts/generate_qa.py",
        env={
            "GOOGLE_APPLICATION_CREDENTIALS": "/opt/airflow/secrets/gcp/service-account.json"
        },
        append_env=True,
    )

    backfill_qa
