from __future__ import annotations

from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.bash import BashOperator

from _dataset_builder import build_bash_command


with DAG(
    dag_id="dataset_builder_clean_corpus",
    description="Refresh the cleaned corpus from raw Parquet inputs.",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "dataset-builder"},
    tags=["dataset-builder", "maintenance"],
) as dag:
    clean_corpus = BashOperator(
        task_id="clean_corpus",
        bash_command=build_bash_command(
            "pipelines/02_clean_corpus.py",
            "--input-dir",
            "data",
            "--output-dir",
            "datasets/clean_corpus",
        ),
    )