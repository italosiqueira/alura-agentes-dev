import rag

perguntas = [
    "O consumidor pode desistir da compra feita pela Internet?",
    "Quais são os direitos do titular de dados pessoais?"
]

for pergunta in perguntas:
    print(f"\n\nPergunta: {pergunta}")

    resposta, contexto = rag.responder_pergunta(pergunta, rerank=True)

    print(f"\nResposta: {resposta}\n")

    for i, doc in enumerate(contexto):
        metadados = dict(filter(lambda pair: pair[0] in ["total_pages", "author", "page", "fonte"], doc.metadata.items()))
        print(f"\n--- Fonte {i+1} (Metadata: {metadados}) ---")
        print(doc.page_content)
