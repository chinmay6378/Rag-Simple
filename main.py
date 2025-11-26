import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint

# ------------------------ Load API Key ------------------------
load_dotenv()

hf_token = (
    st.secrets.get("HF_TOKEN")              # 1. Streamlit Cloud
    or os.getenv("HF_TOKEN")                # 2. .env file
    or os.getenv("HUGGINGFACEHUB_API_TOKEN") # 3. Env variable
)

if not hf_token:
    st.error("❌ HuggingFace API key not found. Add HF_TOKEN in Streamlit secrets or .env file.")
    st.stop()

# ------------------------ UI ------------------------
st.title("📘 Simple One-File RAG App")
st.write("Upload a PDF and ask questions based on its content.")

# ------------------------ LLM ------------------------
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.1,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token  # ← KEY INCLUDED HERE
)

# ------------------------ Embeddings ------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = None


# ------------------------ PDF Upload ------------------------
uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf:
    st.info("Processing PDF...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    vector_db = FAISS.from_documents(chunks, embedding_model)

    st.success("PDF processed successfully! Ask your question below.")


# ------------------------ Ask a Question ------------------------
query = st.text_input("Ask a question from the PDF:")

if st.button("Get Answer"):
    if not vector_db:
        st.error("⚠️ Upload a PDF first.")
    elif query.strip() == "":
        st.warning("Please enter a question.")
    else:
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Use ONLY this context to answer the question.

Context:
{context}

Question: {query}

Answer:
"""

        response = llm.invoke(prompt)

        st.success("Answer:")
        st.write(response)
