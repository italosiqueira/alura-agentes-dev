import os

# Gerenciamento local de chaves de API
import dotenv

# Loaders e chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

# Banco vetorial
from langchain_chroma import Chroma
# Deprecated! The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0
#from langchain_community.vectorstores import Chroma

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Carrega as variáveis de ambiente necessárias (ex: API Keys)
config = dotenv.dotenv_values()

# Pasta do projeto
PROJECT_DIR = os.path.dirname(__file__)

CHROMA_DB_PATH = "chroma_db"

COLECAO_UNIFICADA = "documentos_juridicos"

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

CHUNK_SIZE = 800

CHUNK_OVERLAP = 200

# Embedding model da OpenAI (exemplo: "text-embedding-3-small" ou "text-embedding-3-large")
EMBEDDING_MODEL = "text-embedding-3-small"

embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=config['OPENAI_API_KEY'])

def gerar_chunks_recursivos(documentos, overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=overlap
    )

    return splitter.split_documents(documentos)

def gerar_chunks_paragrafo(documentos, overlap=CHUNK_OVERLAP):
    splitter = CharacterTextSplitter(
        separator="\n\n", 
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=overlap)

    return splitter.split_documents(documentos)

def carregar_documentos():
    """Carrega documentos PDF da pasta 'dados' e adiciona metadados de fonte."""
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

def criar_vectorstore(_persist_directory, _colecao_nome = COLECAO_UNIFICADA):
    
    documentos = carregar_documentos()

    chunks = gerar_chunks_recursivos(documentos)
    
    return Chroma.from_documents(
                documents = chunks,
                embedding = embedding,
                persist_directory= _persist_directory,
                collection_name = _colecao_nome
            )

def carregar_vectorstore(_persist_directory = CHROMA_DB_PATH, _colecao_nome = COLECAO_UNIFICADA):
    chroma_dir = os.path.join(PROJECT_DIR, _persist_directory)
    if os.path.exists(chroma_dir):
        return Chroma(
                    embedding_function=embedding,
                    persist_directory=chroma_dir,
                    collection_name=_colecao_nome
        )

    return criar_vectorstore(chroma_dir)

if __name__ == "__main__":
    chroma_dir = os.path.join(PROJECT_DIR, CHROMA_DB_PATH)
    if os.path.exists(chroma_dir):
        print(f"Pasta do Chroma '{chroma_dir}' existe!")
        vectorstore = carregar_vectorstore(CHROMA_DB_PATH)
        for c in vectorstore._client.list_collections():
            print(
                c.name,
                "→",
                c.count(),
                "documentos"
            )
    else:
        print(f"Pasta do Chroma '{chroma_dir}' não existe! Criando Vectorstore...")
