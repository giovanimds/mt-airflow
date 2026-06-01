CREATE TABLE IF NOT EXISTS public.qa_dataset_quarantine (
    id           UUID PRIMARY KEY,
    -- Colunas originais
    question     TEXT,
    reasoning    TEXT,
    answer       TEXT,
    model        TEXT,
    source       TEXT,
    embedding    FLOAT[],
    metadata     JSONB,
    -- Auditoria
    audit_flags     TEXT[]  NOT NULL DEFAULT '{}',
    audit_scores    JSONB   NOT NULL DEFAULT '{}',
    audit_triggers  TEXT[]  NOT NULL DEFAULT '{}',
    -- Dataset de resiliência
    resposta_desejada  TEXT NOT NULL,
    audited_at         TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_qa_quarantine_flags
    ON qa_dataset_quarantine USING GIN(audit_flags);
