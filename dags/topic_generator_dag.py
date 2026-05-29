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

def _run_topic_generation(**context):
    for p in (SCRIPTS_PATH, DAGS_PATH):
        if p not in sys.path:
            sys.path.insert(0, p)

    from topic_generator import generate_topics, push_topics_to_queue  # noqa: PLC0415

    params = context.get("params", {})
    main_topic = params.get("main_topic", "Inteligência Artificial")
    num_terms = params.get("num_terms", 15)

    log.info(f"Disparando geração manual de tópicos: assunto='{main_topic}', quantidade={num_terms}")
    
    terms = generate_topics(main_topic, num_terms)
    added, duplicates = push_topics_to_queue(terms)
    
    log.info(f"Concluído: {added} novos tópicos inseridos no Valkey, {duplicates} duplicados ignorados.")
    return {"added": added, "duplicates": duplicates, "terms": terms}

with DAG(
    dag_id="topic_generator_dag",
    description="Gera termos de busca de forma assistida por LLM (Mistral) e enfileira no Valkey.",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,  # Apenas acionamento manual
    catchup=False,
    max_active_runs=1,
    params={
        "main_topic": Param(
            "Inteligência Artificial",
            type="string",
            description="Tópico principal para gerar sub-termos de busca.",
        ),
        "num_terms": Param(
            15,
            type="integer",
            minimum=1,
            maximum=50,
            description="Quantidade de sub-termos de busca a serem gerados.",
        ),
    },
    tags=["dataset-builder", "topics", "valkey", "mistral"],
) as dag:

    run_generation = PythonOperator(
        task_id="generate_and_push_topics",
        python_callable=_run_topic_generation,
    )

    run_generation
