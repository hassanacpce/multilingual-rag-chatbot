import os
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

vectordb_file_path = "faiss_index"

# -----------------------------
# LLM
# -----------------------------
# NOTE: llama-3.3-70b-versatile was deprecated by Groq for free/dev-tier usage
# (announced June 17, 2026). Migrated to openai/gpt-oss-120b per Groq's own
# migration guidance. Override via GROQ_TEXT_MODEL if you're on an enterprise
# contract that still serves the old model, or want a different one.
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b"),
    temperature=0
)

# -----------------------------
# Embeddings (FREE)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create Vector DB
# -----------------------------
def create_vector_db():

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
    loader = CSVLoader(
        file_path=dataset_path,
        encoding="utf-8"
    )

    documents = loader.load()

    vectordb = FAISS.from_documents(
        documents,
        embeddings
    )

    vectordb.save_local(vectordb_file_path)

# -----------------------------
# QA Chain
# -----------------------------
def get_qa_chain():

    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(
    search_kwargs={"k": 5}
)

    prompt = PromptTemplate.from_template("""
You are a friendly and professional Customer Service AI Assistant.

Rules:

1. If the user greets you (Hi, Hello, Good Morning, How are you, etc.), reply naturally.
2. If the user asks who you are, introduce yourself.
3. If the question is related to the company or knowledge base, answer ONLY from the provided context.
4. Never make up facts that are not present in the context.
5. If the answer cannot be found in the context, politely reply:

"I'm sorry, I couldn't find that information in my knowledge base."

Context:
{context}

Question:
{input}

Answer:
""")

    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    return chain