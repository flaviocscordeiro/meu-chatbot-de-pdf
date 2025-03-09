import os
import PyPDF2
import glob
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# ================================================
# 1. Configurações gerais
# ================================================
# Substitua pela sua chave da OpenAI ou deixe como variável de ambiente
os.environ["OPENAI_API_KEY"] = "SUA_CHAVE_AQUI"  # se preferir, comente e utilize .env

# ================================================
# 2. Função para ler PDFs e extrair texto
# ================================================
def read_pdfs_from_folder(folder_path):
    """
    Lê todos os arquivos PDF de uma pasta e concatena em uma única string.
    Retorna o texto extraído.
    """
    all_text = ""
    pdf_files = glob.glob(folder_path + "/*.pdf")

    for pdf_file in pdf_files:
        # Abrir cada PDF
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
            all_text += text

    return all_text

# ================================================
# 3. Função para ler TXT e extrair texto
# ================================================
def read_txts_from_folder(folder_path):
    """
    Lê todos os arquivos .txt de uma pasta e concatena em uma única string.
    Retorna o texto extraído.
    """
    all_text = ""
    txt_files = glob.glob(folder_path + "/*.txt")

    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as file:
            all_text += file.read() + "\n"

    return all_text

# ================================================
# 4. Preparar o texto - chunking
# ================================================
def prepare_text(text):
    """
    Divide o texto em chunks menores usando CharacterTextSplitter.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# ================================================
# 5. Construir a base vetorial (Chroma)
# ================================================
def build_vector_db(chunks):
    """
    Gera embeddings usando OpenAIEmbeddings e armazena no Chroma.
    """
    embeddings = OpenAIEmbeddings()  # ou use SentenceTransformerEmbeddings
    vectordb = Chroma.from_texts(chunks, embeddings, collection_name="meus_pdf_chroma")
    return vectordb

# ================================================
# 6. Criar a corrente de perguntas e respostas
# ================================================
def create_qa_chain(vectorstore):
    """
    Cria uma corrente de QA com LangChain e retorna a cadeia conversacional.
    """
    # Modelo de chat (você pode ajustar temperature, etc.)
    llm = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo')

    # Cria um objeto ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        # Defina se quer exibir as fontes, etc.
        return_source_documents=True
    )
    return qa_chain

# ================================================
# 7. Função principal para rodar o chatbot
# ================================================
def main():
    # 7.1. Ler PDFs e TXTs
    folder_path = "inputs"
    pdf_text = read_pdfs_from_folder(folder_path)
    txt_text = read_txts_from_folder(folder_path)
    all_text = pdf_text + txt_text

    if not all_text.strip():
        print("Não foi encontrado texto nos PDFs ou TXT da pasta 'inputs'.")
        return

    # 7.2. Dividir texto em chunks
    chunks = prepare_text(all_text)

    # 7.3. Criar base vetorial
    vector_db = build_vector_db(chunks)

    # 7.4. Criar cadeia de QA
    qa_chain = create_qa_chain(vector_db)

    # 7.5. Loop de perguntas e respostas
    chat_history = []
    print("Chatbot pronto! Faça sua pergunta (ou digite 'sair' para encerrar).")

    while True:
        user_input = input("Você: ")

        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Encerrando o chatbot.")
            break
        
        # Alimentar a cadeia com a pergunta
        result = qa_chain({"question": user_input, "chat_history": chat_history})
        
        # Resposta do bot
        answer = result["answer"]
        source_docs = result["source_documents"]  # Se quiser usar as fontes
        
        # Atualizar histórico (opcional, se quiser usar no prompt)
        chat_history.append((user_input, answer))
        
        print(f"Chatbot: {answer}")
        
        # Se quiser, pode imprimir as fontes relevantes
        # for i, doc in enumerate(source_docs):
        #     print(f"Fonte {i+1}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    main()
