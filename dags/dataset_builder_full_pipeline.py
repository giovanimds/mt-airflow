from __future__ import annotations

from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.bash import BashOperator

from _dataset_builder import build_bash_command


with DAG(
    dag_id="dataset_builder_full_pipeline",
    description="Build the cleaned corpus, Portuguese base dataset, and Simple QA dataset.",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "dataset-builder"},
    tags=["dataset-builder", "batch"],
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

    base_model_pt = BashOperator(
        task_id="build_base_model_pt",
        bash_command=build_bash_command(
            "pipelines/03_base_dataset_pipeline.py",
            "--input-dir",
            "datasets/clean_corpus",
            "--output-dir",
            "datasets/base_model_pt",
        ),
    )

    simple_qa = BashOperator(
        task_id="build_simple_qa",
        bash_command=build_bash_command(
            "pipelines/04_simpleqa_dataset_pipeline.py",
            "--input-dir",
            "datasets/clean_corpus",
            "--output-dir",
            "datasets/simple_qa",
        ),
    )

    clean_corpus >> base_model_pt >> simple_qa