from __future__ import annotations
from datetime import datetime, timezone

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

gcp_volume = k8s.V1Volume(
    name="gcp-service-account",
    secret=k8s.V1SecretVolumeSource(secret_name="airflow-gcp-key"),
)

gcp_volume_mount = k8s.V1VolumeMount(
    name="gcp-service-account",
    mount_path="/opt/airflow/secrets/gcp",
    read_only=True,
)

with DAG(
    dag_id="corpus_cleaner_gemma",
    description="Consome os arquivos raw parquet, limpa com Gemma AI Studio e salva no YugabyteDB.",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule="*/10 * * * *", # Roda a cada 10 minutos para consumir o que estiver na fila
    catchup=False,
    max_active_runs=1, # Apenas uma instância por vez
    default_args={"owner": "dataset-builder"},
    tags=["dataset-builder", "cleaner", "gemma"],
) as dag:

    # Nós rodaremos o corpus_cleaner.py usando a mesma imagem do scraper
    # Ou a imagem do dataset-builder. Vamos usar a do dataset-builder se tiver as libs.
    # Como as libs (google-generativeai, polars, psycopg2) provavelmente estão na do dataset-builder, 
    # usaremos um bash comando para instalar dependências em tempo de execução se for a do scraper,
    # ou exigimos que a imagem tenha. Vamos usar a imagem padrão do Airflow.
    
    cleaner_task = KubernetesPodOperator(
        task_id="run_corpus_cleaner",
        name="corpus-cleaner-pod",
        namespace="airflow",
        image="{{ var.value.get('SCRAPER_IMAGE', 'localhost:5000/corpus-scraper:v1.0') }}",
        image_pull_policy="Always",
        cmds=["python3"],
        arguments=["/app/corpus_cleaner.py"],
        env_vars=[
            k8s.V1EnvVar(
                name="GOOGLE_APPLICATION_CREDENTIALS",
                value="/opt/airflow/secrets/gcp/service-account.json",
            ),
            k8s.V1EnvVar(
                name="REDIS_URL",
                value="{{ var.value.get('REDIS_URL', 'redis://valkey-primary.default.svc.cluster.local:6379') }}",
            ),
            k8s.V1EnvVar(
                name="GEMINI_API_KEY_PAID",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="airflow-api-keys",
                        key="GEMINI_API_KEY_PAID",
                    )
                ),
            ),
            k8s.V1EnvVar(
                name="GEMINI_API_KEY_FREE",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="airflow-api-keys",
                        key="GEMINI_API_KEY_FREE",
                    )
                ),
            ),
            k8s.V1EnvVar(
                name="GEMINI_PAID_MODEL",
                value="gemini-2.5-flash-lite",
            ),
            k8s.V1EnvVar(
                name="PG_HOST",
                value="{{ var.value.get('PG_HOST', 'postgres.morescotech.com.br') }}",
            ),
            k8s.V1EnvVar(
                name="PG_PORT",
                value="5432",
            ),
            k8s.V1EnvVar(
                name="PG_USER",
                value="yugabyte",
            ),
            k8s.V1EnvVar(
                name="PG_PASSWORD",
                value="YugabytePass2026",
            ),
            k8s.V1EnvVar(
                name="PG_DATABASE",
                value="ai_labs",
            ),
        ],
        volumes=[gcp_volume],
        volume_mounts=[gcp_volume_mount],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "200m", "memory": "512Mi"},
            limits={"cpu": "1000m", "memory": "2Gi"},
        ),
    )
