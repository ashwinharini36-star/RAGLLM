import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# ----------------------------------------------------
# Streamlit page config
# ----------------------------------------------------
st.set_page_config(page_title="📄 RAG PDF Summarizer", layout="wide")
st.title("📄 Retrieval-Augmented PDF Q&A")

# ----------------------------------------------------
# Embeddings (local, CPU mode)
# ----------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ----------------------------------------------------
# File uploader
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load and split PDF
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    st.info(f"✅ PDF loaded and split into {len(chunks)} chunks. Creating embeddings...")

    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success("✅ PDF indexed. You can now ask questions.")

    query = st.text_input("Ask a question about your PDF:")

    if query:
        # ----------------------------------------------------
        # Local FLAN-T5 via transformers pipeline
        # ----------------------------------------------------
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=-1  # -1 = CPU, change to 0 if you have GPU
        )
        llm = HuggingFacePipeline(pipeline=generator)

        # Retrieval-QA chain
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        with st.spinner("Thinking..."):
            try:
                answer = qa.run(query)
            except Exception as e:
                st.error("⚠️ Local model call failed.")
                st.write(f"Debug info: {str(e)}")
                st.stop()

        st.subheader("Answer:")
        st.write(answer)
