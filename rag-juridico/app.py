# Importações básicas
import os

# Loader de documentos PDF
from langchain_community.document_loaders import PyPDFLoader

# Divisão de texto em blocos
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
#from langchain_openai import OpenAIEmbeddings

# Banco vetorial
from langchain_community.vectorstores import Chroma

# LLM
#from langchain_openai import ChatOpenAI

# Cadeia RAG
#from langchain.chains import RetrievalQA

def carregar_documentos():
    """Carrega documentos PDF da pasta 'dados' e adiciona metadados de fonte."""
    # Pasta do projeto
    PROJECT_DIR = os.path.dirname(__file__)

    # Pasta 'dados' relativa a este arquivo (usando `os`)
    DATA_DIR = "dados"

    # Lista fixa de documentos na pasta 'dados'
    NOMES = [
        "CDC_2025.pdf",
        "Lei_geral_protecao_dados_pessoais_1ed.pdf",
    ]

    METADADO_FONTE = [
        "cdc", 
        "lgpd"
    ]

    documentos = []

    for i, doc_name in enumerate(NOMES):
        caminho_relativo = os.path.join(PROJECT_DIR, DATA_DIR, doc_name)
        # Valida se o arquivo existe antes de tentar carregá-lo
        if not os.path.isfile(caminho_relativo):
            raise FileNotFoundError(f"PDF not found at: {caminho_relativo}")
        else:
            loader = PyPDFLoader(caminho_relativo)
            docs = loader.load()
            for pagina in docs:
                pagina.metadata["fonte"] = METADADO_FONTE[i]
            documentos.extend(docs)
    
    return documentos


documentos = carregar_documentos()

print(f"Documentos carregados: {len(documentos)}")

# Conta ocorrências para cada fonte usando lambda
fontes = []
fontes = set(list(map(lambda doc: doc.metadata.get("fonte"), documentos)))

for fonte in fontes:
    count = len(list(filter(lambda doc: doc.metadata.get("fonte") == fonte, documentos)))
    print(f"Fonte '{fonte}': {count}")