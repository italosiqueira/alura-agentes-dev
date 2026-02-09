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
from langchain_openai import ChatOpenAI

# Cadeia RAG
#from langchain.chains import RetrievalQA

# Pasta do projeto
PROJECT_DIR = os.path.dirname(__file__)

CHUNK_SIZE = 800

CHUNK_OVERLAP = 200

COLECAO_UNIFICADA = "documentos_juridicos"

# Embedding model da OpenAI (exemplo: "text-embedding-3-small" ou "text-embedding-3-large")
EMBEDDING_MODEL = "text-embedding-3-small"

# Modelo de linguagem da OpenAI (exemplo: "gpt-4o-mini" ou "gpt-4o")
LLM_MODEL = "gpt-4o-mini"

# Carregar a API Key do nosso provedor de modelos de LLMs
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def criar_vectorstore_por_fonte(_chunks):

    # Configuração do Chroma para armazenamento local
    persist_directory = os.path.join(PROJECT_DIR, "chroma_db")
    # Usando um embedding model da OpenAI
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Extrai informações sobre o metadado "fonte"
    fontes = set(list(map(lambda doc: doc.metadata.get("fonte"), _chunks)))

    for fonte in fontes:
        count = len(list(filter(lambda doc: doc.metadata.get("fonte") == fonte, _chunks)))
        print(f"Fonte (chunks) '{fonte}': {count}")
        criar_vectorstore(embeddings, persist_directory, fonte, 
                            list(filter(lambda doc: doc.metadata.get("fonte") == fonte, _chunks)))

def criar_carregar_vectorstore(_chunks, _colecao_nome):
    
    vectorstore = None

    # Configuração do Chroma para armazenamento local
    persist_directory = os.path.join(PROJECT_DIR, "chroma_db")
    # Usando um embedding model da OpenAI
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if (os.path.exists(persist_directory) and os.path.isdir(persist_directory)):
        # Carrega vectorstore pré-existente (não gera cobrança)
        print(f"Carregando VectorStore em {persist_directory}", persist_directory)
        vectorstore = carregar_vectorstore(embeddings, persist_directory, _colecao_nome)
    else:
        # Criação do vectorstore
        print(f"Criando VectorStore em {persist_directory}", persist_directory)
        vectorstore = criar_vectorstore(embeddings, persist_directory, _colecao_nome, _chunks)
    
    print(f"VectorStore '{vectorstore._collection.name}' pronta para uso!")
    return vectorstore

def criar_vectorstore(_embedding, _persist_directory = "./chroma_db", _colecao_nome = "default", _documentos = []):
    return Chroma.from_documents(
        documents = _documentos,
        embedding = _embedding,
        persist_directory= _persist_directory,
        collection_name = _colecao_nome
    )

def carregar_vectorstore(_embedding, _persist_directory = "./chroma_db", _colecao_nome = "default"):
    vectorstore = None
    CHROMA_DB_FILE = "chroma.sqlite3"

    persist_file = os.path.join(_persist_directory, CHROMA_DB_FILE)
    if (os.path.exists(_persist_directory) and os.path.isdir(_persist_directory)
            and os.path.isfile(persist_file)):
        vectorstore = Chroma(
            embedding_function=_embedding,
            persist_directory=_persist_directory,
            collection_name=_colecao_nome
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
# chunks = gerar_chunks_paragrafo(documentos)
print(f"Chunks gerados: {len(chunks)}")
tamanho_medio_chunk = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
print(f"Tamanho médio dos chunks: {tamanho_medio_chunk:.0f} caracteres")

# Extra (opcional): criar coleções separadas por fonte
#criar_vectorstore_por_fonte(chunks)

# Criar coleção unificada
vectorstore = criar_carregar_vectorstore(chunks, COLECAO_UNIFICADA)
for c in vectorstore._client.list_collections():
    print(
        c.name,
        "→",
        c.count(),
        "documentos"
    )

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

perguntas = [
    "O fornecedor pode se eximir de responsabilidade?",
    "Em que casos o consentimento é obrigatório?"
]

for pergunta in perguntas:
    print(f"\n\nPergunta: {pergunta}")
    resultados = retriever.invoke(pergunta)
    for i, doc in enumerate(resultados):
        metadados = dict(filter(lambda pair: pair[0] in ["total_pages", "author", "page", "fonte"], doc.metadata.items()))
        print(f"\n--- Resultado {i+1} (Metadata: {metadados}) ---")
        print(doc.page_content)
