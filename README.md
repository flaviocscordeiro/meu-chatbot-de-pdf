# Meu Chatbot de PDF

Este projeto tem como objetivo facilitar a busca de informações em documentos PDF (ou arquivos .txt) usando um chatbot interativo baseado em IA generativa, embeddings e buscas vetoriais. Com isso, é possível fazer perguntas em linguagem natural e obter respostas relevantes, usando como base o conteúdo dos PDFs selecionados.

## Como funciona?

1. Lemos todos os PDFs e arquivos .txt dentro da pasta `inputs`.
2. Extraímos e dividimos o texto em pedaços (chunks).
3. Criamos embeddings desses chunks e armazenamos em um banco vetorial.
4. Quando o usuário faz uma pergunta, o modelo procura os chunks mais relevantes e gera uma resposta contextualizada.

## Tecnologias utilizadas
- [Python 3.8+](https://www.python.org/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Chroma DB](https://www.trychroma.com/)
- [OpenAI API](https://platform.openai.com/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)

## Requisitos de Instalação
```bash
pip install -r requirements.txt
