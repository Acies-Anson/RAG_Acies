import streamlit as st
import os
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIG ---
CUSTOM_CACHE_DIR = "C:/hf_models" 
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

load_dotenv()
api_key = os.getenv("XAI_API_KEY")

# --- 1. CORE LOGIC ---
def section_aware_splitter(text, filename):
    section_pattern = r'\n(?=\d+\.\s+[A-Z\s]{3,})'
    sections = re.split(section_pattern, text)
    section_docs = []
    for i, content in enumerate(sections):
        lines = content.strip().split('\n')
        header = lines[0] if lines else "INTRO/PREAMBLE"
        sub_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        sub_chunks = sub_splitter.split_text(content)
        for j, chunk in enumerate(sub_chunks):
            section_docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "section": header,
                    "section_no": i,
                    "chunk_no": j + 1,
                    "total_section_chunks": len(sub_chunks)
                }
            ))
    return section_docs

@st.cache_resource
def initialize_rag():
    data_folder = Path("data")
    if not data_folder.exists(): return None, []
    all_chunks = []
    for file_path in data_folder.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            all_chunks.extend(section_aware_splitter(f.read(), file_path.name))
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", cache_folder=CUSTOM_CACHE_DIR)
    vector_db = FAISS.from_documents(all_chunks, embeddings, distance_strategy="COSINE")
    return vector_db, all_chunks

vector_db, all_chunks_list = initialize_rag()

# --- 2. NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Legal Chatbot", "Chunking Inspector"])

# --- PAGE 1: CHATBOT ---
if page == "Legal Chatbot":
    st.title("⚖️ Legal RAG: Conversational Audit")

    if "messages" not in st.session_state: st.session_state.messages = []
    if "last_results" not in st.session_state: st.session_state.last_results = []

    # SIDEBAR: TOP 5 RANKING WITH FULL METADATA
    st.sidebar.header("🔍 Retrieval Ranking (Top 5)")
    for rank, (doc, score) in enumerate(st.session_state.last_results, 1):
        with st.sidebar.expander(f"Rank {rank} | Match: {score*100:.2f}%"):
            st.write(f"**Document:** `{doc.metadata['source']}`")
            st.write(f"**Section:** {doc.metadata['section']}")
            st.write(f"**Section No:** {doc.metadata['section_no']}")
            st.write(f"**Chunk:** {doc.metadata['chunk_no']} / {doc.metadata['total_section_chunks']}")
            st.divider()
            st.caption(doc.page_content)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about the agreements..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Retrieval
        results = vector_db.similarity_search_with_relevance_scores(prompt, k=5)
        st.session_state.last_results = results # Save for the sidebar display
        
        context = "\n\n".join([f"[Source: {d.metadata['source']} | Sec: {d.metadata['section']} | Chunk: {d.metadata['chunk_no']}]: {d.page_content}" for d, s in results])
        
        with st.chat_message("assistant"):
            llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0)
            response = llm.invoke(f"Context: {context}\n\nQuestion: {prompt}\n\nStrictly cite section and chunk IDs.")
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
        st.rerun()

# --- PAGE 2: CHUNKING INSPECTOR ---
else:
    st.title("📦 Chunking & Storage Visualization")
    st.write("This page demonstrates how your legal text was split and stored in the Vector Database.")

    if all_chunks_list:
        # Convert metadata to a DataFrame for clear tabular view
        df = pd.DataFrame([
            {
                "Rank/Index": i,
                "Document": c.metadata["source"],
                "Section": c.metadata["section"],
                "Sec No": c.metadata["section_no"],
                "Chunk No": c.metadata["chunk_no"],
                "Content Preview": c.page_content[:150] + "..."
            } for i, c in enumerate(all_chunks_list)
        ])
        
        st.subheader("1. Metadata Storage Table")
        st.dataframe(df, use_container_width=True)

        st.subheader("2. Visualizing the 'Recursive' Logic")
        st.info("The document was first split by **Section Headings** (Regex), then further divided into sub-chunks (Recursive) to ensure no clause exceeds the LLM's memory limits.")
        
        sample_idx = st.slider("Select a chunk to inspect full content", 0, len(all_chunks_list)-1, 0)
        st.write(f"### Inspecting Chunk {sample_idx}")
        st.json(all_chunks_list[sample_idx].metadata)
        st.code(all_chunks_list[sample_idx].page_content, language="text")