import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# ----------------------------------------------------
# Streamlit page config
# ----------------------------------------------------
st.set_page_config(page_title="📄 RAG PDF Summarizer", layout="wide")

# ----------------------------------------------------
# Hugging Face Token from secrets
# ----------------------------------------------------
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    st.error("⚠️ Hugging Face token not found. Please set it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# ----------------------------------------------------
# Embeddings model
# ----------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.title("📄 Retrieval-Augmented PDF Q&A")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load PDF from temp file
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Build FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success("✅ PDF indexed. You can now ask questions.")

    query = st.text_input("Ask a question about your PDF:")

    if query:
        # Hugging Face Hub model (Flan-T5)
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"max_new_tokens": 512}
        )

        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        with st.spinner("Thinking..."):
            answer = qa.run(query)

        st.subheader("Answer:")
        st.write(answer)
