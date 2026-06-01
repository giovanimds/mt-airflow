from __future__ import annotations
import logging
from datetime import datetime, timezone
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.secret import Secret
from kubernetes.client import models as k8s

log = logging.getLogger(__name__)

AUDITOR_IMAGE = "registry.morescotech.com.br:5000/corpus-scraper:v1.9-auditor"

with DAG(
    dag_id="qa_dataset_auditor",
    description="Audita o qa_dataset em 4 camadas e gera dataset de resiliência usando GPU para NLI",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,  # Apenas acionamento manual
    catchup=False,
    max_active_runs=1,
    tags=["dataset-builder", "auditor", "kubernetes", "gpu"],
) as dag:

    # Mapeamento do segredo da API do Mistral para embeddings
    mistral_api_key = Secret(
        deploy_type="env",
        deploy_target="MISTRAL_API_KEY",
        secret="airflow-api-keys",
        key="MISTRAL_API_KEY",
    )

    qa_auditor_pod = KubernetesPodOperator(
        task_id="qa_auditor_pod",
        name="qa-auditor-pod",
        namespace="airflow",
        image=AUDITOR_IMAGE,
        cmds=["python", "qa_auditor.py"],
        env_vars={
            "PG_HOST": "postgres.morescotech.com.br",
            "PG_PORT": "5432",
            "PG_USER": "yugabyte",
            "PG_PASSWORD": "YugabytePass2026",
            "PG_DATABASE": "ai_labs",
        },
        secrets=[mistral_api_key],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "1000m", "memory": "3Gi", "nvidia.com/gpu": "1"},
            limits={"cpu": "2000m", "memory": "8Gi", "nvidia.com/gpu": "1"},
        ),
        tolerations=[
            k8s.V1Toleration(
                key="nvidia.com/gpu",
                operator="Exists",
                effect="NoSchedule"
            )
        ],
        image_pull_policy="Always",
        startup_timeout_seconds=600,
        get_logs=True,
        is_delete_operator_pod=True,
    )

    qa_auditor_pod
