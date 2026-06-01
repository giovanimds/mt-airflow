import os
import re
import json
import time
import logging
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("qa_auditor")

REGEX_PATTERNS = [
    # Portuguese phantom context patterns
    r"\bno artigo\b",
    r"\bno texto\b",
    r"\bno documento\b",
    r"\bsegundo o autor\b",
    r"\bconforme o texto\b",
    r"\bbaseado no texto\b",
    r"\bde acordo com o texto\b",
    r"\bmencionado no texto\b",
    r"\bcitado no texto\b",
    r"\bsegundo a passagem\b",
    r"\bpassagem fornecida\b",
    r"\bartigo acima\b",
    r"\btexto acima\b",
    r"\bdocumento acima\b",
    r"\binformações fornecidas\b",
    r"\bbaseado no artigo\b",
    r"\bde acordo com o artigo\b",
    r"\bmencionado no artigo\b",
    r"\bconforme o artigo\b",
    r"\bo texto não menciona\b",
    r"\bo artigo não menciona\b",
    r"\bbaseado no documento\b",
    r"\bde acordo com o documento\b",
    r"\bconforme o documento\b",
    r"\bmencionado no documento\b",
    
    # English phantom context patterns
    r"\bin the article\b",
    r"\bin the text\b",
    r"\bin the document\b",
    r"\baccording to the text\b",
    r"\bbased on the text\b",
    r"\bmentioned in the text\b",
    r"\bthe text does not\b",
    r"\bthe article does not\b",
    r"\baccording to the author\b",
    r"\bprovided passage\b",
    r"\bpassage above\b",
    r"\btext above\b",
    r"\barticle above\b",
    r"\bdocument above\b",
    r"\bprovided information\b",
    r"\bbased on the article\b",
    r"\baccording to the article\b",
    r"\bmentioned in the article\b"
]

PRIORITY = ["context_reference", "ungrounded_answer", "hallucination_risk", "semantic_mismatch"]

RESILIENCE = {
    "context_reference": (
        "Olá! Não tenho acesso ao documento mencionado. "
        "Poderia compartilhar o texto para que eu possa te ajudar?"
    ),
    "ungrounded_answer": (
        "Essa pergunta requer um documento de referência específico que "
        "não está disponível para mim agora. Pode me fornecer o contexto?"
    ),
    "hallucination_risk": (
        "Para responder com precisão, preciso de mais contexto ou do "
        "material de referência. Poderia fornecê-lo?"
    ),
    "semantic_mismatch": (
        "Hmm, não tenho certeza se entendi o que está sendo pedido. "
        "Poderia reformular ou fornecer mais detalhes?"
    ),
}

def get_db_connection():
    params = {
        "host": os.environ.get("PG_HOST", "postgres.morescotech.com.br"),
        "port": int(os.environ.get("PG_PORT", 5432)),
        "user": os.environ.get("PG_USER", "yugabyte"),
        "password": os.environ.get("PG_PASSWORD", "YugabytePass2026"),
        "database": os.environ.get("PG_DATABASE", "ai_labs"),
        "sslmode": "disable"
    }
    try:
        return psycopg2.connect(**params, load_balance=True)
    except (TypeError, psycopg2.Error):
        return psycopg2.connect(**params)

def run_transaction_with_retry(get_conn_func, transaction_func, max_retries=10, delay=5):
    for attempt in range(max_retries):
        conn = None
        try:
            conn = get_conn_func()
            with conn:
                with conn.cursor() as cur:
                    result = transaction_func(cur)
            return result
        except psycopg2.Error as e:
            err_str = str(e).lower()
            if any(term in err_str for term in ["catalog version mismatch", "serialization", "mismatched_schema", "ddl occurred"]):
                log.warning(f"Catalog mismatch / serialization error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                if conn:
                    try:
                        conn.rollback()
                        conn.close()
                    except Exception:
                        pass
                time.sleep(delay)
            else:
                if conn:
                    try:
                        conn.rollback()
                        conn.close()
                    except Exception:
                        pass
                raise e
    raise RuntimeError(f"Failed to execute database transaction after {max_retries} attempts due to Catalog Mismatch / Serialization Failure.")

def ensure_quarantine_table():
    log.info("Verificando a existencia da tabela qa_dataset_quarantine...")
    def txn(cur):
        cur.execute("SELECT EXISTS(SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'qa_dataset_quarantine');")
        exists = cur.fetchone()[0]
        if exists:
            log.info("Tabela qa_dataset_quarantine ja existe. Ignorando DDL.")
            return True
        return False
        
    exists = run_transaction_with_retry(get_db_connection, txn)
    if exists:
        return

    log.info("Garantindo a existencia da tabela qa_dataset_quarantine...")
    sql_path = "create_qa_quarantine_table.sql"
    if not os.path.exists(sql_path):
        sql_path = "/app/create_qa_quarantine_table.sql"
    
    if os.path.exists(sql_path):
        with open(sql_path, "r") as f:
            sql = f.read()
    else:
        sql = """
        CREATE TABLE IF NOT EXISTS public.qa_dataset_quarantine (
            id           UUID PRIMARY KEY,
            question     TEXT,
            reasoning    TEXT,
            answer       TEXT,
            model        TEXT,
            source       TEXT,
            embedding    FLOAT[],
            metadata     JSONB,
            audit_flags     TEXT[]  NOT NULL DEFAULT '{}',
            audit_scores    JSONB   NOT NULL DEFAULT '{}',
            audit_triggers  TEXT[]  NOT NULL DEFAULT '{}',
            resposta_desejada  TEXT NOT NULL,
            audited_at         TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_qa_quarantine_flags
            ON qa_dataset_quarantine USING GIN(audit_flags);
        """
    def ddl_txn(cur):
        cur.execute(sql)
    
    run_transaction_with_retry(get_db_connection, ddl_txn)
    log.info("Tabela qa_dataset_quarantine verificada/criada.")


def load_nli_model(device):
    log.info("Carregando modelo NLI (cross-encoder/nli-deberta-v3-small)...")
    model_name = "cross-encoder/nli-deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    log.info("Modelo NLI carregado com sucesso na GPU/CPU.")
    return model, tokenizer

def get_mistral_client():
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY não encontrada no ambiente!")
    base_url = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
    model = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model

def generate_embeddings_batched(client, model, texts, batch_size=32):
    safe_texts = []
    for t in texts:
        if not t or not t.strip():
            safe_texts.append("empty")
        elif len(t) > 24000:
            log.warning(f"Texto muito longo ({len(t)} caracteres). Truncando para 24000 caracteres.")
            safe_texts.append(t[:24000])
        else:
            safe_texts.append(t)

    embeddings = []
    i = 0
    while i < len(safe_texts):
        current_batch_size = min(batch_size, len(safe_texts) - i)
        success = False
        attempts = 0
        while current_batch_size > 0 and not success:
            batch = safe_texts[i:i+current_batch_size]
            try:
                response = client.embeddings.create(model=model, input=batch)
                embeddings.extend([e.embedding for e in response.data])
                i += current_batch_size
                success = True
            except Exception as e:
                err_str = str(e).lower()
                if ("too many tokens" in err_str or "bad request" in err_str or "3210" in err_str or "400" in err_str) and current_batch_size > 1:
                    log.warning(f"Batch with size {current_batch_size} failed with token limit error. Splitting in half...")
                    current_batch_size = current_batch_size // 2
                elif ("exceeding max" in err_str or "too many tokens" in err_str or "3210" in err_str or "400" in err_str) and current_batch_size == 1:
                    log.error(f"Nao foi possivel gerar embedding para item {i} mesmo truncado: {e}. Usando vetor zerado.")
                    embeddings.append([0.0] * 1024)
                    i += 1
                    success = True
                else:
                    attempts += 1
                    log.warning(f"Erro ao gerar embeddings (tentativa {attempts}/5): {e}")
                    if attempts >= 5:
                        raise e
                    time.sleep(2 ** attempts)
    return embeddings

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def run_nli_batched(model, tokenizer, pairs, batch_size, device):
    scores = []
    label2id = getattr(model.config, "label2id", None)
    entail_idx = 2
    if label2id and isinstance(label2id, dict):
        for label, idx in label2id.items():
            if "entail" in label.lower():
                entail_idx = idx
                break
                
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        questions = [p[0] for p in batch]
        answers   = [p[1] for p in batch]
        
        enc = tokenizer(questions, answers,
                        padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = logits.softmax(dim=-1).cpu().numpy()
        scores.extend(probs[:, entail_idx].tolist())
    return scores

def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Dispositivo de processamento: {device}")
    
    ensure_quarantine_table()
    
    # 1. Carrega todos os registros do qa_dataset
    log.info("Buscando registros da tabela qa_dataset...")
    def load_qa_records(cur):
        cur.execute("SELECT id, question, reasoning, answer, model, source, embedding, metadata FROM qa_dataset;")
        return cur.fetchall()
        
    rows = run_transaction_with_retry(get_db_connection, load_qa_records)
    
    qa_records = []
    for r in rows:
        meta = r[7] if isinstance(r[7], dict) else {}
        corpus_id = meta.get("corpus_id")
        qa_records.append({
            "id": r[0],
            "question": r[1],
            "reasoning": r[2],
            "answer": r[3],
            "model": r[4],
            "source": r[5],
            "embedding": r[6],
            "metadata": meta,
            "corpus_id": corpus_id
        })
    log.info(f"Carregados {len(qa_records)} registros do qa_dataset.")
    
    if not qa_records:
        log.info("Nenhum registro encontrado para auditoria. Encerrando.")
        return
        
    # 2. Carrega embeddings de chunks do qa_dataset salvos na tabela embeddings_chunks
    # Para economizar chamadas de API, faremos merge do qa_records['embedding'] com embeddings_chunks
    log.info("Buscando embeddings salvos na tabela embeddings_chunks para qa_dataset...")
    def load_embeddings_chunks(cur):
        cur.execute("SELECT record_id, embedding FROM embeddings_chunks WHERE table_name = 'qa_dataset' AND chunk_index = 0;")
        return cur.fetchall()
        
    chunks_rows = run_transaction_with_retry(get_db_connection, load_embeddings_chunks)
    chunks_embeddings = {row[0]: row[1] for row in chunks_rows}
    
    # Consolidar embeddings da resposta
    missing_answer_ids = []
    missing_answer_texts = []
    
    for r in qa_records:
        emb = r["embedding"]
        if not emb and r["id"] in chunks_embeddings:
            emb = chunks_embeddings[r["id"]]
            r["embedding"] = emb
        if not emb:
            missing_answer_ids.append(r["id"])
            missing_answer_texts.append(r["answer"])
            
    log.info(f"Embeddings de resposta: {len(qa_records) - len(missing_answer_ids)} encontrados no DB, {len(missing_answer_ids)} pendentes.")
    
    # Gerar embeddings ausentes (pergunta e resposta)
    client, embed_model = get_mistral_client()
    
    if missing_answer_ids:
        log.info(f"Gerando {len(missing_answer_ids)} embeddings de resposta ausentes via API Mistral...")
        ans_embs = generate_embeddings_batched(client, embed_model, missing_answer_texts, batch_size=32)
        ans_emb_dict = dict(zip(missing_answer_ids, ans_embs))
        for r in qa_records:
            if r["id"] in ans_emb_dict:
                r["embedding"] = ans_emb_dict[r["id"]]
                
    log.info("Gerando embeddings de pergunta para todos os registros via API Mistral...")
    questions = [r["question"] for r in qa_records]
    q_embs = generate_embeddings_batched(client, embed_model, questions, batch_size=32)
    for r, q_emb in zip(qa_records, q_embs):
        r["question_embedding"] = q_emb
        
    log.info("Embeddings do qa_dataset prontos.")
    
    # 3. Carrega embeddings de corpus do banco de dados
    # Só buscamos corpus_ids que são referenciados no qa_dataset e não são nulos
    corpus_ids = list(set([r["corpus_id"] for r in qa_records if r["corpus_id"]]))
    log.info(f"Carregando embeddings de chunks para {len(corpus_ids)} corpus_ids referenciados...")
    
    corpus_embeddings = {}
    if corpus_ids:
        def load_corpus_embeddings(cur):
            results = []
            batch_size = 5000
            for i in range(0, len(corpus_ids), batch_size):
                sub_ids = corpus_ids[i:i+batch_size]
                cur.execute(
                    "SELECT record_id, embedding FROM embeddings_chunks WHERE table_name = 'corpus' AND record_id = ANY(%s::uuid[]);",
                    (sub_ids,)
                )
                results.extend(cur.fetchall())
            return results
            
        corpus_rows = run_transaction_with_retry(get_db_connection, load_corpus_embeddings)
        for rec_id, emb in corpus_rows:
            if rec_id not in corpus_embeddings:
                corpus_embeddings[rec_id] = []
            corpus_embeddings[rec_id].append(emb)
                    
    log.info(f"Carregados embeddings de chunks para {len(corpus_embeddings)} corpus_ids.")
    
    # 4. Camada 1: Regex
    log.info("Executando Camada 1: Regex...")
    regex_flags = {}
    for r in qa_records:
        matched = []
        q = r["question"] or ""
        a = r["answer"] or ""
        for pattern in REGEX_PATTERNS:
            rx = re.compile(pattern, re.IGNORECASE)
            if rx.search(q) or rx.search(a):
                clean_pat = pattern.replace(r"\b", "")
                matched.append(clean_pat)
        if matched:
            regex_flags[r["id"]] = matched
    log.info(f"Camada 1 concluída. {len(regex_flags)} registros sinalizados.")
    
    # 5. Camada 2: NLI na GPU
    log.info("Executando Camada 2: NLI (DeBERTa)...")
    nli_model, nli_tokenizer = load_nli_model(device)
    pairs = [(r["question"], r["answer"]) for r in qa_records]
    nli_scores_list = run_nli_batched(nli_model, nli_tokenizer, pairs, batch_size=128, device=device)
    nli_scores = dict(zip([r["id"] for r in qa_records], nli_scores_list))
    log.info("Camada 2 concluída.")
    
    # 6. Camada 3: Cosine Q->A
    log.info("Executando Camada 3: Cosine Q->A...")
    cosine_qa_scores = {}
    for r in qa_records:
        q_vec = r["question_embedding"]
        a_vec = r["embedding"]
        sim = cosine_similarity(q_vec, a_vec)
        cosine_qa_scores[r["id"]] = sim
    log.info("Camada 3 concluída.")
    
    # 7. Camada 4: Source Grounding
    log.info("Executando Camada 4: Source Grounding...")
    grounding_scores = {}
    for r in qa_records:
        r_id = r["id"]
        c_id = r["corpus_id"]
        a_vec = r["embedding"]
        
        if not c_id:
            grounding_scores[r_id] = {"score": None, "flag": "source_not_in_corpus"}
            continue
            
        if c_id not in corpus_embeddings:
            # O registro existe no corpus, mas não tem embeddings gerados.
            # Não sinalizamos como source_not_in_corpus para evitar falsos positivos
            grounding_scores[r_id] = {"score": None, "flag": None}
            continue
            
        chunks = corpus_embeddings[c_id]
        sims = [cosine_similarity(a_vec, chunk_vec) for chunk_vec in chunks]
        max_sim = max(sims) if sims else 0.0
        
        flag = "ungrounded_answer" if max_sim < 0.55 else None
        grounding_scores[r_id] = {"score": max_sim, "flag": flag}
    log.info("Camada 4 concluída.")
    
    # 8. Consolidar flags e criar dataset de resiliência
    log.info("Consolidando flags de auditoria...")
    quarantine_records = []
    clean_count = 0
    
    flag_counts = {
        "context_reference": 0,
        "hallucination_risk": 0,
        "semantic_mismatch": 0,
        "ungrounded_answer": 0,
        "source_not_in_corpus": 0
    }
    
    for r in qa_records:
        r_id = r["id"]
        
        # Coletar flags
        flags = []
        triggers = []
        scores = {}
        
        # Regex
        if r_id in regex_flags:
            flags.append("context_reference")
            triggers.extend(regex_flags[r_id])
            
        # NLI
        nli_val = nli_scores[r_id]
        scores["nli_score"] = nli_val
        if nli_val < 0.30:
            flags.append("hallucination_risk")
            triggers.append(f"nli_score:{nli_val:.4f}")
            
        # Cosine Q->A
        cos_qa_val = cosine_qa_scores[r_id]
        scores["cosine_qa"] = cos_qa_val
        if cos_qa_val < 0.50:
            flags.append("semantic_mismatch")
            triggers.append(f"cosine_qa:{cos_qa_val:.4f}")
            
        # Grounding
        g_val = grounding_scores[r_id]
        scores["grounding_score"] = g_val["score"]
        if g_val["flag"] == "source_not_in_corpus":
            flags.append("ungrounded_answer")
            flags.append("source_not_in_corpus")
            triggers.append("source_not_in_corpus")
        elif g_val["flag"] == "ungrounded_answer":
            flags.append("ungrounded_answer")
            triggers.append(f"grounding_score:{g_val['score']:.4f}")
            
        if flags:
            # Dedup flags
            flags = list(set(flags))
            
            # Incrementar estatísticas
            for f in flags:
                if f in flag_counts:
                    flag_counts[f] += 1
            
            # Escolher a resposta de resiliência baseada na prioridade
            chosen_flag = None
            for p_flag in PRIORITY:
                if p_flag in flags:
                    chosen_flag = p_flag
                    break
            if not chosen_flag and "source_not_in_corpus" in flags:
                chosen_flag = "ungrounded_answer"
                
            resposta_desejada = RESILIENCE.get(chosen_flag, RESILIENCE["hallucination_risk"])
            
            quarantine_records.append({
                "id": r["id"],
                "question": r["question"],
                "reasoning": r["reasoning"],
                "answer": r["answer"],
                "model": r["model"],
                "source": r["source"],
                "embedding": r["embedding"],
                "metadata": r["metadata"],
                "audit_flags": flags,
                "audit_scores": scores,
                "audit_triggers": triggers,
                "resposta_desejada": resposta_desejada
            })
        else:
            clean_count += 1
            
    log.info(f"Auditoria concluída. Registros limpos: {clean_count}, Registros sinalizados para quarentena: {len(quarantine_records)}")
    
    # 9. Persistir no banco de dados (inserir na quarentena e excluir do qa_dataset)
    if quarantine_records:
        log.info(f"Movendo {len(quarantine_records)} registros para a quarentena em transação única...")
        move_to_quarantine(quarantine_records)
        log.info("Operação de movimentação concluída com sucesso.")
        
    # 10. Exibir Relatório Final
    t_end = time.time()
    elapsed = t_end - t_start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    
    print("\n" + "="*30 + " RELATÓRIO DE AUDITORIA " + "="*30)
    print(f"Dispositivo:               {device}")
    print(f"Total auditado:            {len(qa_records)}")
    print(f"  Sem flags (mantidos):    {clean_count} ({clean_count/len(qa_records)*100:.2f}%)")
    print(f"  Total -> quarentena:     {len(quarantine_records)} ({len(quarantine_records)/len(qa_records)*100:.2f}%)")
    print("\nDetalhamento de flags (um registro pode ter mais de uma flag):")
    for flag, cnt in flag_counts.items():
        print(f"  - {flag}: {cnt} ({cnt/len(qa_records)*100:.2f}%)")
    print(f"\nTempo total de execução:   {mins}m {secs}s")
    print("="*84 + "\n")

def move_to_quarantine(flagged_records):
    def txn(cur):
        # Inserção
        insert_query = """
            INSERT INTO public.qa_dataset_quarantine (
                id, question, reasoning, answer, model, source, embedding, metadata, 
                audit_flags, audit_scores, audit_triggers, resposta_desejada
            ) VALUES %s
            ON CONFLICT (id) DO NOTHING;
        """
        rows = []
        for r in flagged_records:
            rows.append((
                r["id"],
                r["question"],
                r["reasoning"],
                r["answer"],
                r["model"],
                r["source"],
                r["embedding"],
                json.dumps(r["metadata"]),
                r["audit_flags"],
                json.dumps(r["audit_scores"]),
                r["audit_triggers"],
                r["resposta_desejada"]
            ))
            
        execute_values(cur, insert_query, rows, page_size=100)
        
        # Remoção
        delete_query = "DELETE FROM public.qa_dataset WHERE id = ANY(%s::uuid[]);"
        record_ids = [r["id"] for r in flagged_records]
        cur.execute(delete_query, (record_ids,))
        
    run_transaction_with_retry(get_db_connection, txn)

if __name__ == "__main__":
    main()

