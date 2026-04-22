import streamlit as st
import pandas as pd
from pathlib import Path

# Custom Module Imports
from processor import get_section_chunks
from embeddings import get_embedding_model
from vector_store import create_vector_db
from ranking import get_top_k_results
from core_api import call_llm
from evaluation import evaluate_response

# --- PAGE CONFIG ---
st.set_page_config(page_title="Legal RAG Auditor", layout="wide")

# --- SYSTEM INITIALIZATION ---
@st.cache_resource
def load_system():
    data_folder = Path("data")
    all_chunks = []
    if not data_folder.exists():
        st.error("Data folder not found.")
        return None, []
    
    for file in data_folder.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            all_chunks.extend(get_section_chunks(f.read(), file.name))
    
    embed_model = get_embedding_model()
    db = create_vector_db(all_chunks, embed_model)
    return db, all_chunks

vector_db, all_chunks_list = load_system()

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Chatbot", "Data Inspector"])

if page == "Chatbot":
    st.title("⚖️ Legal Chatbot & Auditor")

    # Initialize Session States
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_results" not in st.session_state:
        st.session_state.last_results = []

    # --- SIDEBAR: RETRIEVAL STATS ---
    if st.session_state.last_results:
        st.sidebar.header("🔍 Current Retrieval Stats")
        for r, (doc, score) in enumerate(st.session_state.last_results, 1):
            with st.sidebar.expander(f"Rank {r} ({score*100:.1f}%)"):
                st.write(f"**Source:** {doc.metadata['source']}")
                st.write(f"**Section:** {doc.metadata['section']}")
                st.caption(f"Chunk: {doc.metadata['chunk_no']}")

    # --- CHAT DISPLAY ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.title(f"{msg['role'].capitalize()}")
            st.markdown(msg["content"])
            # If the message is from assistant and has evaluation data, show it
            if msg["role"] == "assistant" and "evaluation" in msg:
                eval_r = msg["evaluation"]
                st.divider()
                st.caption("📊 **Legal Audit Metrics**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Faithfulness", f"{eval_r['faithfulness_score']}/5")
                c2.metric("Relevance", f"{eval_r['relevance_score']}/5")
                c3.metric("Precision", f"{eval_r['precision_score']}/5")
                st.info(f"**Auditor Feedback:** {eval_r['feedback']}")

    # --- INPUT & LOGIC ---
    if prompt := st.chat_input("Ask a legal question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun() # Rerun to show user message immediately

# This part handles the generation after rerun
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_prompt = st.session_state.messages[-1]["content"]
    
    # 1. Retrieval
    results = get_top_k_results(vector_db, last_prompt)
    st.session_state.last_results = results 
    
    context = "\n\n".join([f"[{d.metadata['section']}]: {d.page_content}" for d, s in results])
    
    with st.chat_message("assistant"):
        # 2. Generate Answer
        response = call_llm(f"Context: {context}\n\nQuestion: {last_prompt}")
        st.markdown(response.content)
        
        # 3. Evaluation Rubric
        with st.spinner("Auditing response quality..."):
            eval_results = evaluate_response(last_prompt, context, response.content)

        # UI Display for current response
        st.divider()
        st.subheader("📊 Performance Rubrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Faithfulness", f"{eval_results['faithfulness_score']}/5")
        col2.metric("Relevance", f"{eval_results['relevance_score']}/5")
        col3.metric("Context Precision", f"{eval_results['precision_score']}/5")
        st.info(f"**Auditor Feedback:** {eval_results['feedback']}")

    # 4. Store Answer + Evaluation in History
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response.content,
        "evaluation": eval_results # This ensures history keeps the scores!
    })
    st.rerun()


# --- PAGE 2: INSPECTOR ---
if page == "Data Inspector":
    st.title("📦 Chunking & Metadata Inspector")
    if all_chunks_list:
        meta_df = pd.DataFrame([c.metadata for c in all_chunks_list])
        st.dataframe(meta_df, use_container_width=True)
    else:
        st.warning("No data indexed yet.")