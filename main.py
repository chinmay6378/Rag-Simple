import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

st.set_page_config(page_title="PDF RAG App", layout="wide")

st.title("📘 AI PDF Question Answering (RAG)")
st.write("Upload a PDF and ask questions based on its content.")

# Embedding Model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.5,
)
chat_model = ChatHuggingFace(llm=llm)

# Global vector DB
vectordb = None
retriever = None


# ------------------ Load PDF Function ------------------
def process_pdf(uploaded_file):
    global vectordb, retriever

    if uploaded_file is None:
        return "⚠️ Please upload a PDF."

    # Save uploaded PDF temporarily
    pdf_path = "temp_uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Create FAISS vectorstore
    vectordb = FAISS.from_documents(chunks, embedding)
    retriever = vectordb.as_retriever()

    return "✅ PDF processed! Now ask your questions."


# ------------------ RAG Answer Function ------------------
def rag_answer(query):
    if retriever is None:
        return "⚠️ Please upload and process a PDF first."

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"Use ONLY this context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = chat_model.invoke(prompt)
    return response.content


# ------------------ Streamlit UI ------------------

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if st.button("Process PDF"):
    msg = process_pdf(uploaded_pdf)
    st.success(msg)

st.divider()

user_query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer = rag_answer(user_query)
        st.text_area("Answer:", answer, height=300)
