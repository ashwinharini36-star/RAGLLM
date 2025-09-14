import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
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
# Embeddings model (via Hugging Face Inference API)
# ----------------------------------------------------
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
)

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

    st.info(f"PDF loaded and split into {len(chunks)} chunks. Creating embeddings in batches...")

    # ----------------------------------------------------
    # Batch embedding with progress bar
    # ----------------------------------------------------
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    batch_size = 16
    all_embeddings = []

    progress = st.progress(0)
    status = st.empty()

    total_batches = (len(texts) - 1) // batch_size + 1

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            st.error(f"⚠️ Embedding batch {i//batch_size+1} failed: {str(e)}")
            st.stop()

        # Update progress
        progress.progress(min((i + batch_size) / len(texts), 1.0))
        status.text(f"Processed batch {i//batch_size + 1} of {total_batches}")

    status.text("✅ All batches processed")

    # Build FAISS vectorstore from embeddings
    vectorstore = FAISS.from_embeddings(all_embeddings, texts, embeddings, metadatas=metadatas)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success("✅ PDF indexed. You can now ask questions.")

    query = st.text_input("Ask a question about your PDF:")

    if query:
        # Hugging Face Hub model (Flan-T5 for answering)
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"max_new_tokens": 512}
        )

        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        with st.spinner("Thinking..."):
            try:
                answer = qa.run(query)
            except Exception as e:
                st.error("⚠️ LLM call failed. Check Hugging Face model or token.")
                st.write(f"Debug info: {str(e)}")
                st.stop()

        st.subheader("Answer:")
        st.write(answer)
