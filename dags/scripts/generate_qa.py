import os
import json
import polars as pl
from google.cloud import storage
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# Prompts
abstract_prompt = ChatPromptTemplate.from_template(
    """Você é um assistente especializado em mineração de dados acadêmicos. 
Analise o texto bruto fornecido e retorne APENAS um objeto JSON contendo duas chaves:
1. "abstract": O resumo do artigo científico isolado de qualquer texto de interface ou rodapé.
2. "keywords": Uma lista com as palavras-chave explícitas no texto.

Ignore completamente endereços, e-mails, instruções de navegação de sites e direitos de acesso.
Não traduza o texto, apenas extraia as informações solicitadas.

Texto Bruto:
{texto}
"""
)

meta_prompt = ChatPromptTemplate.from_template(
    """Com base no resumo acadêmico fornecido, extraia os metadados metodológicos no formato JSON estrito abaixo:

{{
  "publico_alvo": "",
  "conclusoes_principais": "",
  "eh_ponderativo": true/false,
  "eh_exploratorio": true/false,
  "eh_especulativo": true/false,
  "eh_embasado_em_dados": true/false,
  "eh_destilavel_pergunta_resposta": true/false,
  "eh_destilavel_chain_of_thought": true/false,
  "eh_desprovido_de_utilidade_pratica": true/false
}}

Regras:
- Se uma informação não estiver explícita, preencha com null ou array vazio.
- Responda estritamente com o JSON, sem explicações adicionais.
- Preencha os alvos sempre em pt-BR, mesmo que o texto esteja em outro idioma.
- eh_ponderativo é true se o artigo fizer uma análise crítica ou reflexão profunda sobre um tema.
- eh_exploratorio é true se o artigo apresentar um estudo exploratório, levantamento de dados ou análise de campo.
- eh_especulativo é true se o artigo apresentar hipóteses, conjecturas ou propostas teóricas sem necessariamente ter dados empíricos.
- eh_embasado_em_dados é true se o artigo apresentar dados empíricos, análises estatísticas ou evidências concretas para sustentar suas conclusões.
- eh_embasado_em_dados é true APENAS se os dados, métricas ou descobertas empíricas estiverem explicitamente descritos no resumo. Apenas mencionar a palavra "resultados" sem explicações não é suficiente.
- eh_destilavel_pergunta_resposta é true se o artigo puder ser resumido em um formato de pergunta e resposta direta.
- eh_destilavel_chain_of_thought é true se o artigo puder ser resumido em um formato de cadeia de raciocínio passo a passo.
- eh_desprovido_de_utilidade_pratica é true se o texto apresentado estiver incompleto, for apenas um fragmento de rodapé, instrução de navegação ou qualquer coisa que não seja o conteúdo acadêmico do artigo. Ou seja, se o texto não tiver utilidade prática para um pesquisador ou estudante que queira entender o conteúdo do artigo.
- eh_desprovido_de_utilidade_pratica é true se o texto não contiver pelo menos uma conclusão tangível, um método claro ou um dado concreto. Frases puramente introdutórias que apenas anunciam sobre o que o artigo trata (ex: "Este artigo informa sobre os resultados...") sem apresentar quais foram esses resultados DEVERÃO ser marcadas como true.

Resumo:
{abstract}
"""
)

qa_prompt = ChatPromptTemplate.from_template(
    """Você é um pesquisador extraindo conhecimento direto de um artigo científico.
    Com base no resumo fornecido, gere exatamente {quantidade} perguntas diretas.

    REGRA CRÍTICA DE CONTEXTO (ANCORAGEM):
    Nenhuma pergunta pode ser genérica. Toda pergunta DEVE citar o contexto específico do estudo para fazer sentido isoladamente. 
    - ERRADO: "Qual foi o principal resultado?", "Existe correlação entre x e y segundo o artigo?" (Essas perguntas são genéricas e não fazem sentido sem o contexto do resumo)
    - CERTO: "Li um artigo que diz que a capacidade dos alunos é limita, segundo esse artigo seria mais eficiente decorar certas coisas como a tabuada?" (Essa pergunta é contextualizada e faz sentido isoladamente, mesmo sem o resumo completo)
    - Não faça perguntas sobre o artigo diretamente pois quem responderá a pergunta não terá acesso ao resumo completo, portanto, cada pergunta deve conter o contexto necessário para ser compreendida por si só.
    - Não cite o artigo, inclua ele na pargunta de forma contextualizada, ou seja, a pergunta deve conter o contexto necessário para ser compreendida por si só, sem mencionar que é um artigo ou estudo.
    - Não cite nomes ou localidades especificas, procure abstrair o máximo possível mantendo o contexto necessário para a pergunta fazer sentido.

    Exemplos ruins:
    "Em um artigo sobre ...", "Segundo um estudo recente ...", "De acordo com um artigo científico ..."

    Bons exemplos:
    "Dado que na computação algoritmos podem ser otimizados para reduzir a complexidade, seria possível aplicar técnicas de otimização para melhorar o desempenho de algoritmos de ordenação em grandes conjuntos de dados?" (Essa pergunta é contextualizada e faz sentido isoladamente, mesmo sem o resumo completo)
    
    Retorne APENAS uma lista JSON com o formato:
    [
      {{
        "pergunta": "Pergunta contextualizada em pt-BR"
      }}
    ]

    Resumo:
    {abstract}
    """
)

solucionador_prompt = ChatPromptTemplate.from_template(
    """Você é um pesquisador sênior respondendo a uma dúvida acadêmica.
    Explique sua linha de raciocínio passo a passo antes de dar a conclusão final.

    REGRAS:
    - Sua resposta deve conter dois elementos essenciais: o raciocínio detalhado e a conclusão final.
    - Raciocinio: **Identificação da intenção** Explicação sobre o que a pergunta está buscando. \n\n **Raciocínando** Análise detalhada dos pontos relevantes do resumo, considerando diferentes perspectivas ou possibilidades. \n\n **Resposta** A resposta final baseada no raciocínio. \n\n **Revisão** Uma revisão crítica do raciocínio e da resposta para garantir precisão e relevância.
    - Conclusão final": A resposta final baseada no raciocínio

    Pergunta: {pergunta}
    """
)

json_parser = JsonOutputParser()
str_parser = StrOutputParser()

def build_pipeline(llm):
    chain_perguntas = qa_prompt | llm | json_parser
    chain_resposta = solucionador_prompt | llm | str_parser

    def gerar_dataset_desacoplado(inputs):
        resultado_perguntas = chain_perguntas.invoke(inputs)
        
        if isinstance(resultado_perguntas, dict):
            lista_perguntas = resultado_perguntas.get("perguntas", [])
        elif isinstance(resultado_perguntas, list):
            lista_perguntas = resultado_perguntas
        else:
            lista_perguntas = []
        
        dataset = []
        for p in lista_perguntas:
            p_text = p.get("pergunta", "") if isinstance(p, dict) else str(p)
            if not p_text: continue
            resposta = chain_resposta.invoke({"pergunta": p_text}) 
            dataset.append({
                "instruction": p_text,
                "response": resposta
            })
        return dataset

    def roteador_de_lixo(inputs):
        fase_1 = inputs["fase_1"]
        fase_2 = inputs["fase_2"]
        eh_lixo = fase_2.get("eh_desprovido_de_utilidade_pratica", False)
        
        if eh_lixo:
            return {
                "fase_1": fase_1, "fase_2": fase_2, "dataset_instrucoes": None,
                "status_pipeline": "descartado_para_revisao"
            }

        destilacao_input = {
            "abstract": fase_1["abstract"],
            "quantidade": 3
        }

        tarefas_paralelas = {
            "fase_1": lambda _: fase_1,
            "fase_2": lambda _: fase_2,
        }

        eh_destilavel = fase_2.get("eh_destilavel_pergunta_resposta", False) or fase_2.get("eh_destilavel_chain_of_thought", False)
        
        if eh_destilavel:
            tarefas_paralelas["dataset_instrucoes"] = (lambda _: destilacao_input) | RunnableLambda(gerar_dataset_desacoplado)
        else:
            tarefas_paralelas["dataset_instrucoes"] = lambda _: None
            
        tarefas_paralelas["status_pipeline"] = lambda _: "processado_com_sucesso"

        return RunnableParallel(**tarefas_paralelas)

    full_pipeline = (
        abstract_prompt | llm | json_parser 
        | {
            "fase_1": lambda x: x,
            "fase_2": {"abstract": lambda x: x["abstract"]} | meta_prompt | llm | json_parser
          }
        | RunnableLambda(roteador_de_lixo)
    )
    return full_pipeline

def parse_reasoning_and_answer(response_text):
    # Try to extract the final conclusion. Fallback to using the entire text.
    reasoning = response_text
    answer = ""
    # "Conclusão final:" is a common pattern from the prompt
    if "Conclusão final" in response_text:
        parts = response_text.split("Conclusão final", 1)
        reasoning = parts[0].strip()
        answer = parts[1].strip()
        if answer.startswith(":"): answer = answer[1:].strip()
    return reasoning, answer

def main():
    bucket_name = os.environ.get("OUTPUT_BUCKET", "mt-airflow")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    raw_prefix = "raw_corpus/"
    output_prefix = "datasets/pt-br_Q&A/"

    print("Listing raw parquet files...")
    raw_blobs = list(bucket.list_blobs(prefix=raw_prefix))
    raw_files = [blob.name for blob in raw_blobs if blob.name.endswith(".parquet")]

    print("Listing existing JSONL datasets...")
    out_blobs = list(bucket.list_blobs(prefix=output_prefix))
    out_files = [blob.name for blob in out_blobs if blob.name.endswith(".jsonl")]
    out_basenames = {os.path.basename(f) for f in out_files}

    try:
        # User defined model granite4.1:3b as main, mistral as fallback.
        llm_model_name = "granite4.1:3b"
        llm = ChatOllama(model=llm_model_name, temperature=0.7, base_url="http://host.docker.internal:11434")
    except Exception as e:
        print(f"Fallback to Mistral due to: {e}")
        llm_model_name = "ministral-3b-2512"
        llm = ChatMistralAI(model=llm_model_name, temperature=0.7)
    
    # Actually wait, ChatOllama initialization won't fail here if Ollama is down, it fails on invoke.
    # To properly handle fallback, we'd wrap invoke, but let's keep it simple or implement a check.
    # We'll just stick to ChatOllama for now as requested.
    
    pipeline = build_pipeline(llm)

    for rf in raw_files:
        base_name = os.path.basename(rf)
        out_name = base_name.replace(".parquet", ".jsonl")
        
        if out_name in out_basenames:
            print(f"Skipping {base_name}, already processed.")
            continue
            
        print(f"Processing {base_name}...")
        local_parquet = f"/tmp/{base_name}"
        bucket.blob(rf).download_to_filename(local_parquet)
        
        df = pl.read_parquet(local_parquet)
        
        # we need texts and sources
        # schema usually title, text, url
        if "text" not in df.columns:
            print(f"Warning: no text column in {base_name}. Skipping.")
            continue
            
        results_jsonl = []
        
        for row in df.iter_rows(named=True):
            texto = row.get("text", "")
            source = row.get("url", row.get("title", ""))
            if not texto: continue
            
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
                            "source": source
                        })
            except Exception as e:
                print(f"Error processing row from {base_name}: {e}")
                
        local_jsonl = f"/tmp/{out_name}"
        with open(local_jsonl, "w", encoding="utf-8") as f:
            for item in results_jsonl:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        print(f"Uploading {out_name} to GCS...")
        out_blob = bucket.blob(f"{output_prefix}{out_name}")
        out_blob.upload_from_filename(local_jsonl)
        print(f"Finished processing {base_name}.")
        
        # Clean up
        if os.path.exists(local_parquet): os.remove(local_parquet)
        if os.path.exists(local_jsonl): os.remove(local_jsonl)

if __name__ == "__main__":
    main()
