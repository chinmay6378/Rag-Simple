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

# Initialize session variables ONLY once
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Embedding + LLM
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", temperature=0.5)
chat_model = ChatHuggingFace(llm=llm)


# ------------------ Load PDF Function ------------------
def process_pdf(uploaded_file):
    if uploaded_file is None:
        return False, "⚠️ Please upload a PDF."

    try:
        pdf_path = "temp_uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # Store inside session_state
        st.session_state.vectordb = FAISS.from_documents(chunks, embedding)
        st.session_state.retriever = st.session_state.vectordb.as_retriever()

        return True, f"✅ PDF processed successfully! {len(chunks)} chunks created."

    except Exception as e:
        return False, f"❌ Error: {str(e)}"


# ------------------ RAG Answer Function ------------------
def rag_answer(query):
    if st.session_state.retriever is None:
        return False, "⚠️ Please upload and process a PDF first."

    try:
        docs = st.session_state.retriever.get_relevant_documents(query)
        if len(docs) == 0:
            return False, "⚠️ No relevant content found in the PDF."

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"Use ONLY this context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = chat_model.invoke(prompt)

        return True, response.content

    except Exception as e:
        return False, f"❌ Error generating answer: {str(e)}"


# ------------------ UI ------------------

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if st.button("Process PDF"):
    success, msg = process_pdf(uploaded_pdf)
    if success:
        st.success(msg)
    else:
        st.error(msg)

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

