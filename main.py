import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# Load secrets
HF_TOKEN = st.secrets["HF_TOKEN"]
HF_USERNAME = st.secrets["HF_USERNAME"]

st.set_page_config(page_title="PDF RAG App", layout="wide")
st.title("📘 AI PDF Question Answering (RAG)")
st.write("Upload a PDF and ask questions based on its content.")

# Initialize session variables
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# CPU embedding model (Streamlit-friendly)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# LLM (HuggingFace inference)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN,
)

chat_model = ChatHuggingFace(llm=llm)

# PDF processing
def process_pdf(uploaded_file):
    if uploaded_file is None:
        return False, "⚠️ Please upload a PDF."

    try:
        pdf_path = "uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        st.session_state.vectordb = FAISS.from_documents(chunks, embedding)
        st.session_state.retriever = st.session_state.vectordb.as_retriever()

        return True, f"✅ PDF processed successfully! {len(chunks)} chunks created."

    except Exception as e:
        return False, f"❌ Error: {str(e)}"

# RAG querying
def rag_answer(query):
    if st.session_state.retriever is None:
        return False, "⚠️ Please upload and process a PDF first."

    try:
        docs = st.session_state.retriever.invoke(query)

        if not docs:
            return False, "⚠️ No relevant content found."

        context = "\n\n".join([d.page_content for d in docs])

        prompt = (
            f"Use ONLY the context below to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n"
            f"ANSWER:"
        )

        response = chat_model.invoke(prompt)
        return True, response.content

    except Exception as e:
        return False, f"❌ Error generating answer: {type(e).__name__}: {str(e)}"

# UI
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if st.button("Process PDF"):
    success, msg = process_pdf(uploaded_pdf)
    st.success(msg) if success else st.error(msg)

st.divider()

user_query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            success, response = rag_answer(user_query)

        if success:
            st.text_area("Answer:", response, height=300)
        else:
            st.error(response)
