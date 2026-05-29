from __future__ import annotations
import logging
from datetime import datetime, timezone
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import sys
import os

# Adiciona scripts/ ao path
_local_scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if _local_scripts_path not in sys.path:
    sys.path.insert(0, _local_scripts_path)

from model_utils import fetch_models_from_apis, get_db_connection

log = logging.getLogger(__name__)

def sync_models_to_db():
    """Fetches models from APIs and updates the YugabyteDB table."""
    log.info("Fetching models from external APIs...")
    models = fetch_models_from_apis()
    if not models:
        raise ValueError("No models fetched from APIs. This might indicate a network or configuration issue.")

    log.info("Connecting to YugabyteDB to sync %d models...", len(models))
    conn = get_db_connection()
    if not conn:
        raise ConnectionError("Could not connect to YugabyteDB. Check your credentials and network settings.")

    try:
        with conn.cursor() as cur:
            # Upsert logic for YugabyteDB/Postgres
            for provider, model_id in models:
                log.debug("Syncing model: %s: %s", provider, model_id)
                cur.execute("""
                    INSERT INTO available_models (provider, model_id, last_seen)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (provider, model_id) 
                    DO UPDATE SET last_seen = EXCLUDED.last_seen;
                """, (provider, model_id))
            
        conn.commit()
        log.info("Successfully synced %d models to DB.", len(models))
    except Exception as e:
        log.error("Error during model sync: %s", e)
        conn.rollback()
        raise
    finally:
        conn.close()

with DAG(
    dag_id="sync_models_dag",
    description="Atualiza a lista de modelos disponíveis no YugabyteDB",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule="@hourly",
    catchup=False,
    tags=["maintenance", "yugabyte"],
) as dag:

    sync_task = PythonOperator(
        task_id="sync_models",
        python_callable=sync_models_to_db,
    )
