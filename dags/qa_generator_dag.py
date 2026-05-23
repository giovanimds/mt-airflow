from __future__ import annotations

from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectsWithPrefixExistenceSensor

GCS_BUCKET = "mt-airflow"
RAW_PREFIX = "raw_corpus/"
OUT_PREFIX = "datasets/pt-br_Q&A/"


def _find_and_process_new_chunks(**context):
    """
    Identifica os chunks de Parquet que ainda não foram processados e
    executa a geração de Q&A apenas para os novos.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/dags/scripts")
    sys.path.insert(0, "/opt/airflow/dags")

    from google.cloud import storage
    from scripts.generate_qa import build_pipeline, parse_reasoning_and_answer
    import polars as pl
    import json
    import os
    from langchain_ollama import ChatOllama
    from langchain_mistralai import ChatMistralAI

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    # Lista os parquet brutos e os jsonl já gerados
    raw_blobs = [b.name for b in bucket.list_blobs(prefix=RAW_PREFIX) if b.name.endswith(".parquet")]
    out_blobs = {
        os.path.basename(b.name)
        for b in bucket.list_blobs(prefix=OUT_PREFIX)
        if b.name.endswith(".jsonl")
    }

    pending = [rf for rf in raw_blobs if os.path.basename(rf).replace(".parquet", ".jsonl") not in out_blobs]

    if not pending:
        print("Nenhum chunk novo encontrado. Nada a fazer.")
        return

    print(f"{len(pending)} chunk(s) novo(s) encontrado(s): {pending}")

    # Inicializa o modelo
    try:
        llm_model_name = "granite4.1:3b"
        llm = ChatOllama(model=llm_model_name, temperature=0.7, base_url="http://localhost:11434")
    except Exception as e:
        print(f"Fallback para Mistral: {e}")
        llm_model_name = "ministral-3b-2512"
        llm = ChatMistralAI(model=llm_model_name, temperature=0.7)

    pipeline = build_pipeline(llm)

    for rf in pending:
        base_name = os.path.basename(rf)
        out_name = base_name.replace(".parquet", ".jsonl")
        local_parquet = f"/tmp/{base_name}"
        local_jsonl = f"/tmp/{out_name}"

        print(f"Processando {base_name}...")
        bucket.blob(rf).download_to_filename(local_parquet)
        df = pl.read_parquet(local_parquet)

        if "text" not in df.columns:
            print(f"Sem coluna 'text' em {base_name}. Pulando.")
            continue

        results_jsonl = []
        for row in df.iter_rows(named=True):
            texto = row.get("text", "")
            source = row.get("url", row.get("title", ""))
            if not texto:
                continue
            try:
                res = pipeline.invoke({"texto": texto})
                if res.get("status_pipeline") == "processado_com_sucesso" and res.get("dataset_instrucoes"):
                    for qa in res["dataset_instrucoes"]:
                        reasoning, answer = parse_reasoning_and_answer(qa["response"])
                        results_jsonl.append({
                            "question": qa["instruction"],
                            "reasoning": reasoning,
                            "answer": answer,
                            "model": llm_model_name,
                            "source": source,
                        })
            except Exception as e:
                print(f"Erro na linha de {base_name}: {e}")

        with open(local_jsonl, "w", encoding="utf-8") as f:
            for item in results_jsonl:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        bucket.blob(f"{OUT_PREFIX}{out_name}").upload_from_filename(local_jsonl)
        print(f"✅ {out_name} salvo no bucket com {len(results_jsonl)} exemplos.")

        if os.path.exists(local_parquet):
            os.remove(local_parquet)
        if os.path.exists(local_jsonl):
            os.remove(local_jsonl)


with DAG(
    dag_id="qa_generator_dag",
    description=(
        "Monitora o bucket GCS e gera Q&A para cada novo chunk de corpus. "
        "Roda a cada 30 min; processa apenas arquivos ainda não convertidos."
    ),
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    # Verifica a cada 30 minutos se há novos chunks
    schedule="*/30 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "dataset-builder"},
    tags=["dataset-builder", "qa", "generation", "sensor"],
) as dag:

    # Sensor: aguarda ao menos 1 parquet no prefixo raw_corpus/
    # Se não houver nenhum arquivo, encerra sem erro (soft_fail=True)
    aguardar_novos_chunks = GCSObjectsWithPrefixExistenceSensor(
        task_id="aguardar_novos_chunks",
        bucket=GCS_BUCKET,
        prefix=RAW_PREFIX,
        google_cloud_conn_id="google_cloud_default",
        mode="reschedule",       # libera o worker enquanto aguarda
        poke_interval=120,       # verifica a cada 2 min dentro do ciclo
        timeout=25 * 60,         # desiste após 25 min (< schedule de 30 min)
        soft_fail=True,          # não falha o DAG se o bucket estiver vazio
    )

    processar_novos_chunks = PythonOperator(
        task_id="processar_novos_chunks",
        python_callable=_find_and_process_new_chunks,
    )

    aguardar_novos_chunks >> processar_novos_chunks
