# Importações básicas
import os

import ingestao as bd

# Gerenciamento local de chaves de API
from dotenv import load_dotenv

# Loaders e chunking
from langchain_community.document_loaders import PyPDFLoader

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Banco vetorial
from langchain_chroma import Chroma
# Deprecated! The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0
#from langchain_community.vectorstores import Chroma

# LLM
from langchain_openai import ChatOpenAI

# Cadeia RAG
#from langchain.chains import RetrievalQA

# Prompt
from langchain_core.prompts import PromptTemplate

# Pasta do projeto
PROJECT_DIR = os.path.dirname(__file__)

COLECAO_UNIFICADA = "documentos_juridicos"

# Embedding model da OpenAI (exemplo: "text-embedding-3-small" ou "text-embedding-3-large")
EMBEDDING_MODEL = "text-embedding-3-small"

# Modelo de linguagem da OpenAI (exemplo: "gpt-4o-mini" ou "gpt-4o")
LLM_MODEL = "gpt-4o-mini"

# Carregar a API Key do nosso provedor de modelos de LLMs
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        ).content

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

def responder_pergunta(pergunta, vectorstore, rerank=False):
    # LLM
    llm = ChatOpenAI(
        model=LLM_MODEL, 
        temperature=0.0
    )

    # Recuperando o retriever do Vectorstore (opcional)
    # É possivel utilizar o Vectorstore diretamente
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 15}
    )

    # Recuperação inicial via retriever (top-15)
    documentos_recuperados = retriever.invoke(pergunta)

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

# Criar coleção unificada
vectorstore = bd.carregar_vectorstore()

perguntas = [
    "O consumidor pode desistir da compra feita pela Internet?",
    "Quais são os direitos do titular de dados pessoais?"
]

for pergunta in perguntas:
    print(f"\n\nPergunta: {pergunta}")

    resposta, contexto = responder_pergunta(pergunta, vectorstore, True)

    print(f"\nResposta: {resposta}\n")

    for i, doc in enumerate(contexto):
        metadados = dict(filter(lambda pair: pair[0] in ["total_pages", "author", "page", "fonte"], doc.metadata.items()))
        print(f"\n--- Fonte {i+1} (Metadata: {metadados}) ---")
        print(doc.page_content)
