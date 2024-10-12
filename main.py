from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM

from langchain_aws import BedrockEmbeddings

import os


PROMPT_TEMPLATE = """You are a helpful assistant that helps people find information.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
---
ONLY answer questions based on the context above.
Question: {question}
"""

CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")

def get_embedding_func():
    embeddings=BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",  # example model
        region_name="eu-central-1"
    )
    return embeddings
db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_func()
        )

def load_doc(file_path: str):
    doc_loader = PyPDFLoader(file_path)
    return doc_loader.load()

def split_doc(docs:list[Document]):
    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return doc_splitter.split_documents(docs)



def add_to_vectorstore(chunks:list[Document]):

    filtered_chunks = [doc for doc in chunks if doc.page_content and doc.page_content.strip()]
    db.add_documents(filtered_chunks)
    # db.persist()


def query_rag(query_str: str):
    db_results = db.similarity_search(query_str, k=20)
    # print("##############",db_results)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format_messages(context=db_results, question=query_str)

    model = OllamaLLM(model="gemma3")
    response = model.invoke(prompt)
    print("LLM answers here: ", response)


def main():
    print("Hello from easy-rag APP!")
    docs = load_doc("data/TheLittlePrince.pdf")
    add_to_vectorstore(docs)
    
    while True:
        user_input = input("Enter something (or 'quit' to exit): ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        print(f"You asked: {user_input}")
        print(query_rag(user_input))
    
    # print(split_doc(docs)[3])


if __name__ == "__main__":
    main()
