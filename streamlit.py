import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- WINDOWS PATH FIX ---
# Forces the heavy model files into a short path to avoid WinError 206
CUSTOM_CACHE_DIR = "C:/hf_models" 
if not os.path.exists(CUSTOM_CACHE_DIR):
    try:
        os.makedirs(CUSTOM_CACHE_DIR)
    except:
        CUSTOM_CACHE_DIR = "./local_cache" # Fallback to project dir

# --- PAGE CONFIG ---
st.set_page_config(page_title="Data Protection RAG", layout="wide")
st.title("📄 Data Protection Assistant (Groq)")

# 1. Setup & Environment
load_dotenv()
api_key = os.getenv("XAI_API_KEY")

# --- DATA PROCESSING ---
@st.cache_resource
def initialize_rag():
    data_folder = Path("data")
    if not data_folder.exists():
        return None, "Folder 'data' not found."

    documents = []
    for file_path in data_folder.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append(Document(
                page_content=f.read(), 
                metadata={"source": file_path.name}
            ))

    if not documents:
        return None, "No .txt files found."

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    
    # We use the CUSTOM_CACHE_DIR to bypass the WinError 206
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder=CUSTOM_CACHE_DIR
    )
    vector_db = FAISS.from_documents(chunks, embeddings, distance_strategy="COSINE")
    
    return vector_db, f"Successfully indexed {len(documents)} documents."

vector_db, status_msg = initialize_rag()

# --- SIDEBAR: METADATA & RANKING ---
st.sidebar.header("🔍 Retrieval Metadata")
if vector_db is None:
    st.sidebar.error(status_msg)
else:
    st.sidebar.success(status_msg)

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []

# Display previous metadata in sidebar
for rank, (doc, score) in enumerate(st.session_state.last_results, 1):
    st.sidebar.info(f"**Rank {rank}** | Match: {score*100:.2f}%\n\nSource: {doc.metadata['source']}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about your policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval
    results = vector_db.similarity_search_with_relevance_scores(prompt, k=3)
    st.session_state.last_results = results # Save for sidebar
    
    context_chunks = []
    for doc, score in results:
        context_chunks.append(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}")
    
    # LLM Generation
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0)
            
            context_text = "\n\n".join(context_chunks)
            full_prompt = f"""Use the context to answer the question. Cite sources.
            Context: {context_text}
            Question: {prompt}"""
            
            response = llm.invoke(full_prompt)
            st.markdown(response.content)
            
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    st.rerun() 