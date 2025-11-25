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

# Embedding model — instantiate once
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# LLM (HuggingFace endpoint)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.5,
)
chat_model = ChatHuggingFace(llm=llm)

# Globals
vectordb = None
retriever = None

# ------------------ Load PDF Function ------------------
def process_pdf(uploaded_file):
    global vectordb, retriever

    if uploaded_file is None:
        return False, "⚠️ Please upload a PDF."

    try:
        # Save uploaded PDF temporarily
        pdf_path = "temp_uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        if not docs or len(docs) == 0:
            return False, "⚠️ No pages found in PDF (loader returned 0 docs)."

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        if not chunks or len(chunks) == 0:
            return False, "⚠️ No chunks created from PDF (splitter returned 0 chunks)."

        # Create FAISS vectorstore — this can raise if embeddings dim mismatch
        vectordb = FAISS.from_documents(chunks, embedding)
        retriever = vectordb.as_retriever()

        return True, f"✅ PDF processed! {len(chunks)} chunks indexed."

    except Exception as e:
        # Return full exception text so you can debug the real cause
        return False, f"❌ Error while processing PDF: {type(e).__name__}: {str(e)}"


# ------------------ RAG Answer Function ------------------
def rag_answer(query):
    global retriever, chat_model, llm

    if retriever is None:
        return False, "⚠️ Please upload and process a PDF first."

    try:
        # Prefer standard retriever method names depending on version:
        docs = None
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        elif hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(query)
        elif hasattr(retriever, "get_relevant_items"):
            docs = retriever.get_relevant_items(query)
        else:
            # last fallback: try calling as function (some retrievers are callable)
            try:
                docs = retriever(query)
            except Exception:
                raise RuntimeError("Retriever does not expose known retrieval methods.")

        if not docs or len(docs) == 0:
            return False, "⚠️ No documents retrieved for this query."

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"Use ONLY this context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # call the chat model — different wrappers return different shapes, handle common patterns:
        resp_text = None
        # 1) if ChatHuggingFace exposes .invoke (some wrappers do)
        if hasattr(chat_model, "invoke"):
            resp = chat_model.invoke(prompt)
            # many wrappers place text under .content or .text or .response
            resp_text = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
        # 2) some wrappers use .generate([...]) returning an object with .generations
        elif hasattr(chat_model, "generate"):
            gen = chat_model.generate([prompt])
            try:
                resp_text = gen.generations[0][0].text
            except Exception:
                # defensive fallback
                resp_text = str(gen)
        # 3) fallback to direct llm if available
        elif hasattr(llm, "invoke"):
            resp = llm.invoke(prompt)
            resp_text = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
        else:
            raise RuntimeError("Chat model does not expose known invocation methods.")

        if not resp_text:
            resp_text = "⚠️ The model returned an empty response."

        return True, resp_text

    except Exception as e:
        return False, f"❌ Error while answering: {type(e).__name__}: {str(e)}"


# ------------------ Streamlit UI ------------------
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
            success, result = rag_answer(user_query)
        if success:
            st.text_area("Answer:", result, height=300)
        else:
            st.error(result)
