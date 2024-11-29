import gradio as gr
import requests
import os
from dotenv import load_dotenv
import git
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

# Load environment variables
load_dotenv()

# Backend API URL
API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
LOCAL_REPO_DIR = "./repos"


def clone_repository(repo_url):
    """Clone a repository to a local directory."""
    try:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(LOCAL_REPO_DIR, repo_name)
        if os.path.exists(repo_path):
            return f"Repository '{repo_name}' already cloned."
        git.Repo.clone_from(repo_url, repo_path)
        return f"Repository '{repo_name}' cloned successfully!"
    except Exception as e:
        return f"Failed to clone repository: {e}"


def index_repository(repo_path):
    """Index the repository content for RAG."""
    try:
        documents = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith((".md", ".txt")):
                    file_path = os.path.join(root, file)
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            openai_api_type=os.getenv("OPENAI_API_KEY")
        )  # Substitute with your embedding model if needed
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        return f"Failed to index repository: {e}"


def rag_query(vector_store, query):
    """Perform a RAG query on indexed content."""
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(), retriever=retriever, return_source_documents=True
        )

        # Use `invoke()` instead of `run()`
        response = qa_chain.invoke({"query": query})

        # Extract both the result (answer) and source_documents
        result = response.get("result", "No answer found.")
        source_documents = response.get("source_documents", [])

        # Optionally format the source documents for display
        formatted_sources = "\n".join([doc.page_content for doc in source_documents])

        return f"Answer: {result}\n\nSources:\n{formatted_sources}"

    except Exception as e:
        return f"Failed to perform RAG query: {e}"


def handle_query(repo_url, query):
    """Handle repository cloning, indexing, and querying."""
    clone_status = clone_repository(repo_url)
    if "Failed" in clone_status:
        return clone_status

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(LOCAL_REPO_DIR, repo_name)
    vector_store = index_repository(repo_path)
    if isinstance(vector_store, str):
        return vector_store  # Return error message if indexing failed

    return rag_query(vector_store, query)


def create_ui():
    """Build the Gradio UI."""
    with gr.Blocks() as demo:
        gr.Markdown("## RAG-Powered Codebase Chat App")

        with gr.Row():
            repo_url_input = gr.Textbox(
                label="GitHub Repository URL", placeholder="Enter repo URL to clone"
            )
            query_input = gr.Textbox(
                label="Your Question", placeholder="Ask a question about the repo"
            )
            query_button = gr.Button("Submit Query")
            output = gr.Textbox(label="Response", interactive=False)

        query_button.click(
            fn=lambda repo_url, query: handle_query(repo_url, query),
            inputs=[repo_url_input, query_input],
            outputs=[output],
        )

    return demo


if __name__ == "__main__":
    os.makedirs(LOCAL_REPO_DIR, exist_ok=True)
    demo = create_ui()
    demo.launch()
