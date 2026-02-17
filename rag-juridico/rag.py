# Importações básicas
import os

import ingestao as bd

# Gerenciamento local de chaves de API
import dotenv

# LLM
from langchain_openai import ChatOpenAI

# Cadeia RAG
#from langchain.chains import RetrievalQA

# Prompt
from langchain_core.prompts import PromptTemplate

# Carrega as variáveis de ambiente necessárias (ex: API Keys)
config = dotenv.dotenv_values()

# Modelo de linguagem da OpenAI (exemplo: "gpt-4o-mini" ou "gpt-4o")
LLM_MODEL = "gpt-4o-mini"

# LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0, api_key=config['OPENAI_API_KEY'])

# Recupera os documentos na Vectorstore
vectorstore = bd.carregar_vectorstore()

# Recuperando o retriever do Vectorstore (opcional)
# Para uso com a cadeia RAG da LangChain, o retriever é necessário. 
# Mas como estamos implementando a cadeia manualmente, podemos usar 
# diretamente o vectorstore para recuperação.
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}
)

def rerank_documentos(pergunta, documentos, llm):
    
    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"], 
        template="""
Você é um especialista em Direito do Consumidor e Lei Geral de Proteção de Dados (LGPD).

Pergunta do usuário:
{pergunta}

Trecho do documento:
{texto}
    
Avalie a relevância desse trecho para responder a pergunta.
Atribua uma nota entre 0 e 10, onde 0 significa "não é relevante" e 10 significa "extremamente relevante".

Responda apenas com um número de 0 a 10.
"""
    )

    documentos_com_score = []

    for doc in documentos:
        score = llm.invoke(
            prompt_rerank.format(
                pergunta=pergunta,
                texto=doc.page_content
            )
        ).content.strip()

        try:
            score = float(score)
        except:
            score = 0.0
        
        documentos_com_score.append((score, doc))

    documentos_ordenados = sorted(
        documentos_com_score, 
        key=lambda pair: pair[0], 
        reverse=True
    )

    return [doc for _, doc in documentos_ordenados]

def responder_pergunta(pergunta, rerank=False):

    # Recuperação inicial via Chroma Vectorstore (top-15)
    documentos_recuperados = vectorstore.similarity_search(pergunta, k=15)

    contexto_final = []
    if (rerank):
        # ReRanking
        documentos_rerankeados = rerank_documentos(
            pergunta, 
            documentos_recuperados, 
            llm
        )
        contexto_final = documentos_rerankeados[:4]
    else:    
        contexto_final = documentos_recuperados[:4]

    contexto_texto = "\n\n".join(
        [doc.page_content for doc in contexto_final]
    )

    prompt_final = f"""
Responda somente com base no contexto fornecido.

Contexto:
{contexto_texto}

Pergunta:
{pergunta}
"""
    resposta = llm.invoke(prompt_final)
    
    return resposta.content, contexto_final