<div align="center" >
  <h1>CONECTA+ | Agentes de IA para DEVs | Projeto Sprint 01</h1>
</div>

<div style="display: inline_block" align="center" >
	<img alt="Python 3.12.10" src="https://img.shields.io/badge/python-3.12.10-gray?style=flat-square&logo=python&labelColor=brightgreen" />
  <img alt="Langchain" src="https://img.shields.io/badge/langchain-%231C3C3C?style=flat-square&logo=langchain" />
</div>

# Projeto de RAG Jurídico
Este projeto tem como objetivo desenvolver um sistema de RAG (Retrieval-Augmented Generation) para consultas ao CDC - Código de Defesa do Consumidor - e à LGPD - Lei Geral de Proteção de Dados - utilizando a biblioteca LangChain, a API da OpenAI e o banco vetorial Chroma.

Este é o projeto de exercício para a _Sprint 01_ da trilhe de aprendizado _CONECTA+ 2026 - Agentes de IA para DEVs_.

# Montagem do ambiente de desenvolvimento

Crie uma pasta dedicada ao projeto. Em seguida, execute os seguintes comandos a partir desta nova pasta:

```bash
# 1. Cria e ativa um ambiente de desenvolvimento virtual
python -m venv .venv
source venv/bin/activate # ou .venv\Scripts\activate no Windows

# 2. Clonar o projeto a partir do repositório (Git deve estar instalado)
git clone https://github.com/italosiqueira/alura-agentes-dev.git

# 3. Instalar as dependências a partir do requirements.txt do projeto
python -m pip install -r requirements.txt
```

## Solucionador de problemas

### Permissões para execução de código de módulos no Windows

No ambiente Windows, caso você se depare com um erro informando que a execução de um módulo foi bloqueada pelo sistema, tente adicionar a pasta onde os módulos ou arquivos DLL das bibliotecas Python são guardados à variável PATH do SO. No _Powershell_ seria esse o comando:

```powershell
$newPath="C:\<projeto>\.venv\Lib\site-packages";$p=[Environment]::GetEnvironmentVariable("PATH","User");if($p -notlike "*$newPath*"){[Environment]::SetEnvironmentVariable("PATH",$p+";"+$newPath,"User")}
```

**projeto** é a pasta onde o seu projeto está localizado.

## Estrutura do projeto
- `ingestao.py`: arquivo responsável pela criação e gerenciamento da base vetorial utilizando a biblioteca Chroma, incluindo a função de indexação dos documentos jurídicos.
- `rag.py`: arquivo principal contendo a implementação do sistema de RAG, incluindo a configuração do modelo de linguagem, a base vetorial e as funções de recuperação e geração de respostas.
- `app.py`: arquivo contendo a implementação de uma interface simples para interação com o sistema de RAG, permitindo que os usuários façam perguntas e recebam respostas baseadas nos documentos jurídicos indexados.
- `dados/`: pasta contendo os arquivos PDF do CDC e da LGPD, que serão processados e indexados na base vetorial.
- `.env.exemplo`: arquivo de exemplo para configuração das variáveis de ambiente, como a chave da API da OpenAI e o caminho dos dados.
