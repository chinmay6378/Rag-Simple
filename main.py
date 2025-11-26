import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


HF_TOKEN = st.secrets["HF_TOKEN"]

st.set_page_config(page_title="PDF RAG App", layout="wide")
st.title("📘 AI PDF Question Answering (RAG)")
st.write("Upload a PDF and ask questions based on its content.")

# Session state
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Remote embedding model (No Torch, No GPU, No local load)
embedding = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN
)
chat_model = ChatHuggingFace(llm=llm)

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

def rag_answer(query):
    if st.session_state.retriever is None:
        return False, "⚠️ Please upload and process a PDF first."

    try:
        docs = st.session_state.retriever.invoke(query)

        if not docs:
            return False, "⚠️ No relevant content found."

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"Use ONLY this context:\n{context}\n\nQuestion: {query}\nAnswer:"

        result = chat_model.invoke(prompt)
        return True, result.content

    except Exception as e:
        return False, f"❌ Error generating answer: {type(e).__name__}: {str(e)}"

# UI
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if st.button("Process PDF"):
    success, msg = process_pdf(uploaded_pdf)
    st.success(msg) if success else st.error(msg)

st.divider()

question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            success, answer = rag_answer(question)

        if success:
            st.text_area("Answer:", answer, height=300)
        else:
            st.error(answer)
