from __future__ import annotations

from datetime import datetime, timezone
from airflow import DAG
from airflow.operators.bash import BashOperator
from _dataset_builder import build_bash_command

with DAG(
    dag_id="generate_multiturn_conversations",
    description="Generate 100k multi-turn conversational datasets using Gemini Model Pool (flash models).",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "dataset-builder"},
    tags=["dataset-builder", "gemini-pool", "conversational"],
) as dag:
    
    generate_conversations = BashOperator(
        task_id="generate_conversations",
        bash_command=build_bash_command(
            "pipelines/06_generate_multiturn_conversations.py",
            "--output-dir",
            "datasets/conversational_multiturn",
            "--num-conversations",
            "100000",
        ),
    )

    generate_conversations
