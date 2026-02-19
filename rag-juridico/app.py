import rag

def efetua_pergunta():
    print('---\n')
    return input('# PERGUNTA (digite "sair" para encerrar) \n')


def imprime_fontes(fontes: dict[str, list[str]]):
    print('## FONTES:')

    for i, doc in enumerate(fontes):
        metadados = dict(filter(lambda pair: pair[0] in ["total_pages", "author", "page", "fonte"], doc.metadata.items()))
        print(f'  - Fonte {i+1} (Metadata: {metadados})')

def inicia_chat():
    print('### BEM-VINDO AO ASSISTENTE JURÍDICO! FAÇA SUAS PERGUNTAS SOBRE O CDC E A LGPD. ###')
    
    prompt = efetua_pergunta()
    
    while prompt.strip().lower() != 'sair':
        resposta, contexto = rag.responder_pergunta(prompt, rerank=True)
        print(f'\n# RESPOSTA\n{resposta}\n')
        
        if (resposta.strip() != "Desculpe, só posso responder perguntas sobre o CDC e a LGPD."):
            imprime_fontes(contexto)

        prompt = efetua_pergunta()

    print('### OBRIGADO POR USAR O ASSISTENTE JURÍDICO! ATÉ LOGO! ###')


if __name__ == "__main__":
    inicia_chat()