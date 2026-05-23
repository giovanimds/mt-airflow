from __future__ import annotations

from datetime import datetime, timezone
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="qa_generator_dag",
    description="Gera exemplos Q&A a partir do corpus bruto (Parquet -> JSONL).",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "dataset-builder"},
    tags=["dataset-builder", "qa", "generation"],
) as dag:

    # Define the bash command to run the script.
    # Assuming dags are mapped to /opt/airflow/dags inside the container.
    # If this runs locally, the path will just be relative to airflow home, but using a robust path or just standard python execution is better.
    # Airflow typically mounts the dags folder in $AIRFLOW_HOME/dags.
    
    generate_qa_task = BashOperator(
        task_id="generate_qa",
        bash_command="python /opt/airflow/dags/scripts/generate_qa.py || python dags/scripts/generate_qa.py",
        env={
            # Emulating standard Airflow GCP credential paths if needed, 
            # though it might already be in environment
            "GOOGLE_APPLICATION_CREDENTIALS": "/opt/airflow/secrets/gcp/service-account.json"
        },
        append_env=True
    )

    generate_qa_task
