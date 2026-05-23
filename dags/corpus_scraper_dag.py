from __future__ import annotations

from datetime import datetime, timezone
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# Configurações do Kubernetes Pod
# O volume e volumeMount correspondem ao segredo criado para as credenciais da GCP
gcp_volume = k8s.V1Volume(
    name="gcp-service-account",
    secret=k8s.V1SecretVolumeSource(secret_name="airflow-gcp-sa"),
)

gcp_volume_mount = k8s.V1VolumeMount(
    name="gcp-service-account",
    mount_path="/opt/airflow/secrets/gcp",
    read_only=True,
)

with DAG(
    dag_id="raw_corpus_scraper",
    description="Crawl multiple sources (PT-BR), chunk into 50 documents, and save as Parquet in GCS.",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "dataset-builder"},
    tags=["dataset-builder", "scraper", "corpus"],
) as dag:

    spiders = ["wikipedia_pt", "reddit_pt", "gutenberg_pt"]
    
    for spider_name in spiders:
        KubernetesPodOperator(
            task_id=f"run_{spider_name}_scraper",
            name=f"{spider_name.replace('_', '-')}-scraper-pod",
            namespace="airflow",
            # Imagem que contém o script scraper.py
            image="{{ var.value.get('SCRAPER_IMAGE', 'localhost:5000/corpus-scraper:v1.0') }}",
            image_pull_policy="Always",
            # Passagem de variáveis de ambiente configuráveis para o container do Scrapy
            env_vars=[
                k8s.V1EnvVar(
                    name="GOOGLE_APPLICATION_CREDENTIALS",
                    value="/opt/airflow/secrets/gcp/service-account.json",
                ),
                k8s.V1EnvVar(
                    name="OUTPUT_BUCKET",
                    value="{{ var.value.get('SCRAPER_OUTPUT_BUCKET', 'mt-airflow') }}",
                ),
                k8s.V1EnvVar(
                    name="MAX_DOCUMENTS",
                    value="{{ var.value.get('SCRAPER_MAX_DOCUMENTS', '500000') }}",
                ),
                k8s.V1EnvVar(
                    name="CHUNK_SIZE",
                    value="50",
                ),
                k8s.V1EnvVar(
                    name="SPIDER_NAME",
                    value=spider_name,
                ),
            ],
            volumes=[gcp_volume],
            volume_mounts=[gcp_volume_mount],
            # Definição de limites e requisições de recursos para evitar OOM no cluster
            container_resources=k8s.V1ResourceRequirements(
                requests={"cpu": "200m", "memory": "256Mi"},
                limits={"cpu": "1000m", "memory": "1Gi"},
            ),
        )
