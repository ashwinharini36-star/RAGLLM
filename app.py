import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# --- Load FAISS index ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# --- Local model pipeline (runs without API calls) ---
generator = pipeline("text2text-generation", model="google/flan-t5-base")
llm = HuggingFacePipeline(pipeline=generator)

# --- Retrieval Q&A chain ---
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Streamlit UI ---
st.title("📊 Annual Report Q&A (Local Pipeline)")

query = st.text_input("Ask a question about the annual report:")
if query:
    answer = qa.run(query)
    st.write("**Answer:**", answer)
