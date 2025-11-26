import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

load_dotenv()

# ------------------ STREAMLIT SETTINGS ------------------
st.set_page_config(page_title="PDF RAG App", layout="wide")
st.title("📘 AI PDF Question Answering (RAG)")
st.write("Upload a PDF and ask questions with AI!")

# ------------------ LOAD HF TOKEN ------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

# ------------------ SESSION STATE ------------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ------------------ EMBEDDINGS ------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ------------------ LLM ------------------
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN,   # IMPORTANT FIX
)

chat_model = ChatHuggingFace(llm=llm)

# ------------------ PROCESS PDF ------------------
def process_pdf(uploaded_file):
    if uploaded_file is None:
        return False, "⚠️ Please upload a PDF."

    try:
        # Save locally
        pdf_path = "temp_uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load doc
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        # Vector DB
        vectordb = FAISS.from_documents(chunks, embedding)
        retriever = vectordb.as_retriever()

        # Save in session
        st.session_state.vectordb = vectordb
        st.session_state.retriever = retriever

        return True, f"✅ PDF processed! {len(chunks)} chunks generated."

    except Exception as e:
        return False, f"❌ Error: {str(e)}"

# ------------------ RAG ANSWER ------------------
def rag_answer(query):
    if st.session_state.retriever is None:
        return False, "⚠️ Please upload and process a PDF first."

    try:
        retriever = st.session_state.retriever
        docs = retriever.invoke(query)

        if not docs:
            return False, "⚠️ No relevant content found."

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question: {query}

Answer:
"""

        response = chat_model.invoke(prompt)
        return True, response.content

    except Exception as e:
        return False, f"❌ Error generating answer: {type(e).__name__}: {str(e)}"


# ------------------ STREAMLIT UI ------------------

uploaded_pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])

if st.button("Process PDF"):
    success, msg = process_pdf(uploaded_pdf)
    if success:
        st.success(msg)
    else:
        st.error(msg)

st.divider()

user_query = st.text_input("Ask a question from the PDF:")

if st.button("Get Answer"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            success, answer = rag_answer(user_query)

        if success:
            st.text_area("Answer:", answer, height=300)
        else:
            st.error(answer)
