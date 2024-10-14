# RAG CLI

This is a command-line application that allows you to ask questions about a PDF document and get answers from a large language model. The application uses a Retrieval Augmented Generation (RAG) architecture to provide answers based on the content of the document.

## How it works

The application takes a PDF document as input, splits it into smaller chunks, and then uses a language model to generate embeddings for each chunk. The embeddings are stored in a Chroma vector store. When you ask a question, the application searches the vector store for the most relevant chunks of text and then uses a large language model to generate an answer based on the retrieved context.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up your environment:**

    The application uses AWS Bedrock for embeddings and Ollama for the language model. You will need to have an AWS account and have the AWS CLI configured with your credentials. You will also need to have Ollama installed and running.

3.  **Add your document:**

    Place the PDF document you want to query in the `data` directory. The application is currently configured to use `TheLittlePrince.pdf`.

## Usage

1.  **Run the application:**

    ```bash
    python main.py
    ```

2.  **Ask a question:**

    Once the application is running, you can enter your questions at the prompt. To exit the application, type `quit`, `exit`, or `q`.
