import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

# ----------------------
# Config
# ----------------------
st.set_page_config(page_title="RAG PDF Summarizer", layout="wide")

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------
# Streamlit UI
# ----------------------
st.title("📄 RAG PDF Summarizer")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Load and split PDF
    loader = PyPDFLoader(uploaded_file)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Build FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success("✅ PDF indexed. You can now ask questions.")

    query = st.text_input("Ask a question about your PDF:")

    if query:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Initialize small answering model (Flan-T5 from Hugging Face Hub)
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0, "max_length": 512}
        )

        from langchain.chains import RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        with st.spinner("Thinking..."):
            answer = qa.run(query)

        st.write("### Answer:")
        st.write(answer)
