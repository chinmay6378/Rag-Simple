import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
import tempfile

st.title("📘 Simple One-File RAG App")
st.write("Upload a PDF and ask questions based on its content.")

# ------------------------ LLM ------------------------
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.1,
    max_new_tokens=256
)

# ------------------------ Embeddings ------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = None  # global state after upload


# ------------------------ PDF Upload ------------------------
uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf:
    st.info("Processing PDF... Please wait.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = text_splitter.split_documents(pages)

    # Create vector store
    vector_db = FAISS.from_documents(chunks, embedding_model)

    st.success("PDF processed successfully! You can now ask questions.")


# ------------------------ Ask a Question ------------------------
query = st.text_input("Ask a question based on your PDF:")

if st.button("Get Answer"):
    if not vector_db:
        st.error("Please upload a PDF first.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        st.info("Searching...")

        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Use ONLY the following context to answer the question.

Context:
{context}

Question: {query}

Answer:
"""

        answer = llm.invoke(prompt)

        st.success("Answer:")
        st.write(answer)
