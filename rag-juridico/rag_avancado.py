import os

# Gerenciamento local de chaves de API
import dotenv

# Biblioteca para acesso à nossa Vectorstore
import ingestao as bd

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Pasta do projeto
PROJECT_DIR = os.path.dirname(__file__)

# Modelo de linguagem da OpenAI (exemplo: "gpt-4o-mini" ou "gpt-4o")
LLM_MODEL = "gpt-4o-mini"

# Carrega as variáveis de ambiente necessárias (ex: API Keys)
dotenv.load_dotenv(os.path.join(PROJECT_DIR, ".env"))
config = dotenv.dotenv_values()

# Recupera os documentos na Vectorstore
vectorstore = bd.carregar_vectorstore()

# Modelo utilizado para rescrever a pergunta do usuário
rewriter_model = ChatOpenAI(model=LLM_MODEL, temperature=0.0, api_key=config['OPENAI_API_KEY'])

rewriter_prompt_template = """
Você é um assistente especializado em reescrever perguntas de usuários humanos a fim de serem mais efetivas para a recuperação de documentos.
Reescreva a seguinte pergunta do usuário para ser mais efetiva na recuperação de documentos relevantes. Se necessário, adicione um pouco mais de contexto para que a pergunta seja mais clara, mas sem alterar o seu sentido origina.

Pergunta do usuário: {pergunta}
Consulta revisada do Vector DB:
"""

# O PromptTemplate é uma versão simplificada quando não estivermos lidando com um cenário interativo. Nossa solicitação é direta.
rewriter_prompt = PromptTemplate.from_template(rewriter_prompt_template)

rewriter_query_chain = rewriter_prompt | rewriter_model | StrOutputParser()

# Modelo utilizado para ReRanking de documentos
reranking_model = ChatOpenAI(model=LLM_MODEL, temperature=0.0, api_key=config['OPENAI_API_KEY'])

# Prompt template para ReRanking de documentos
reranking_prompt = PromptTemplate(
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

# ReRanking Chain: PromptTemplate + LLM + OutputParser
reranking_query_chain = reranking_prompt | reranking_model | StrOutputParser();

rag_model = reranking_model

rag_prompt_template = """
Responda somente com base no contexto fornecido.
Se a pergunta não estiver relacionada ao CDC ou à LGPD, responda exatamente esta frase: "Desculpe, só posso responder perguntas sobre o CDC e a LGPD."

Contexto:
{contexto}

Pergunta:
{pergunta}
"""

rag_prompt = PromptTemplate(
    input_variables=["contexto", "pergunta"],
    template=rag_prompt_template
)

rag_chain = rag_prompt | rag_model | StrOutputParser()

def rankeia_documentos(prompt_original):
    # Recupera os documentos mais relevantes para a pergunta original
    documentos_relevantes = vectorstore.similarity_search(prompt_original, k=10)

    documentos_relevantes_scored = []

    for doc in documentos_relevantes:
        score = reranking_query_chain.invoke(
                    {"pergunta": prompt_original, "texto":doc.page_content}
                )

        try:
            score = float(score)
        except:
            score = 0.0
        
        documentos_relevantes_scored.append((score, doc))

    documentos_relevantes_scored_ordered_desc = sorted(
        documentos_relevantes_scored, 
        key=lambda pair: pair[0], 
        reverse=True
    )

    return [doc for _, doc in documentos_relevantes_scored_ordered_desc]

def rewrite_query(prompt_original):
    return rewriter_query_chain.invoke(prompt_original)

def responder_pergunta(pergunta, rerank=False):
    # Reescreve a pergunta do usuário para ser mais efetiva na recuperação de documentos relevantes
    pergunta_reescrita = rewrite_query(pergunta)

    print(f'\n# PERGUNTA REVISADA: {pergunta_reescrita}\n')

    # Recupera os documentos mais relevantes para a pergunta reescrita
    if rerank:
        documentos_relevantes = rankeia_documentos(pergunta_reescrita)[:5]
    else:
        documentos_relevantes = vectorstore.similarity_search(pergunta_reescrita, k=5)

    # Monta o contexto a partir dos documentos recuperados
    contexto = "\n\n".join([doc.page_content for doc in documentos_relevantes])

    resposta = rag_chain.invoke(
        {"contexto": contexto, "pergunta": pergunta_reescrita}
    )

    return resposta, documentos_relevantes

if __name__ == "__main__":
    print(rewrite_query("Quais são os direitos do consumidor em relação a produtos com defeito?"))