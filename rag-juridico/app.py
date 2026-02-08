# Importações básicas
import os

# Gerenciamento local de chaves de API
from dotenv import load_dotenv

# Loaders e chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Divisão de texto em blocos
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Banco vetorial
from langchain_chroma import Chroma
# Deprecated! The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0
#from langchain_community.vectorstores import Chroma

# LLM
#from langchain_openai import ChatOpenAI

# Cadeia RAG
#from langchain.chains import RetrievalQA

# Pasta do projeto
PROJECT_DIR = os.path.dirname(__file__)

CHROMA_DB_FILE = "chroma.sqlite3"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"

# Carregar a API Key do nosso provedor de modelos de LLMs
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def criar_vectorstore(_chunks):
    """Cria um vectorstore usando Chroma a partir dos chunks e embeddings fornecidos."""
    vectorstore = None

    # Configuração do Chroma para armazenamento local
    persist_directory = os.path.join(PROJECT_DIR, "chroma_db")
    print(f"Pasta da VectorStore Chroma: {persist_directory}")

    persist_file = os.path.join(PROJECT_DIR, "chroma_db", CHROMA_DB_FILE)
    collection_name = "documentos_juridicos"
    if (os.path.exists(persist_file) and os.path.isfile(persist_file)):
        print(f"Arquivo '{persist_file}' já existe. Carregando VectorStore...")
        vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    else:
        # Criação do vectorstore
        print(f"Criando VectorStore...")
        vectorstore = Chroma.from_documents(
            documents=_chunks,
            # Usando um embedding model da OpenAI
            embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    
    return vectorstore

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


chunks = gerar_chunks_recursivos(documentos)
print(f"Chunks gerados: {len(chunks)}")
tamanho_medio_chunk = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
print(f"Tamanho médio dos chunks: {tamanho_medio_chunk:.0f} caracteres")

# chunks = gerar_chunks_paragrafo(documentos)
# print(f"Chunks gerados: {len(chunks)}")
# tamanho_medio_chunk = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
# print(f"Tamanho médio dos chunks: {tamanho_medio_chunk:.0f} caracteres")

vectorstore = criar_vectorstore(chunks)
for c in vectorstore._client.list_collections():
    print(
        c.name,
        "→",
        c.count(),
        "documentos"
    )
    exemplo = c.get(limit=5)
    for meta in exemplo["metadatas"]:
        print(f"  - {meta}")
