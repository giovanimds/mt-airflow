from __future__ import annotations
import logging
from datetime import datetime, timezone
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.secret import Secret
from kubernetes.client import models as k8s

log = logging.getLogger(__name__)

CLEANER_IMAGE = "192.168.0.7:5000/corpus-scraper:v1.2"

with DAG(
    dag_id="corpus_cleaner_dag",
    description="Executa o Corpus Cleaner como um Pod Kubernetes todas as manhãs",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule="0 6 * * *",  # Todas as manhãs às 6:00
    catchup=False,
    max_active_runs=1,
    tags=["dataset-builder", "cleaner", "kubernetes"],
) as dag:

    # Mapeamento do segredo para o modelo grátis
    gemini_free_key = Secret(
        deploy_type="env",
        deploy_target="GEMINI_API_KEY_FREE",
        secret="airflow-api-keys",
        key="GEMINI_API_KEY_FREE",
    )

    clean_corpus_pod = KubernetesPodOperator(
        task_id="clean_corpus_pod",
        name="corpus-cleaner-pod",
        namespace="airflow",
        image=CLEANER_IMAGE,
        cmds=["python", "corpus_cleaner.py"],
        env_vars={
            "REDIS_URL": "redis://:246608@valkey-primary.default.svc.cluster.local:6379",
            "PG_HOST": "postgres.morescotech.com.br",
            "PG_PORT": "5432",
            "PG_USER": "yugabyte",
            "PG_PASSWORD": "YugabytePass2026",
            "PG_DATABASE": "ai_labs",
        },
        secrets=[gemini_free_key],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "200m", "memory": "512Mi"},
            limits={"cpu": "1000m", "memory": "2Gi"},
        ),
        image_pull_policy="Always",
        get_logs=True,
        is_delete_operator_pod=True,
    )

    clean_corpus_pod
