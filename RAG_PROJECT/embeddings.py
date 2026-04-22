import os
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    # Local directory to cache the downloaded model
    # Avoids re-downloading from HuggingFace on every run
    cache_dir = "C:/hf_models"
    
    # Create the cache folder if it doesn't already exist
    # exist_ok=True prevents error if folder is already there
    os.makedirs(cache_dir, exist_ok=True)
    
    return HuggingFaceEmbeddings(
        # BGE-small-en-v1.5 by Beijing Academy of AI
        # Specifically trained for retrieval tasks using contrastive learning
        # Produces 384-dimensional vectors — balances speed and accuracy
        # Better retrieval scores than MiniLM on MTEB benchmark (~62 vs ~57)
        model_name="BAAI/bge-small-en-v1.5",
        
        # Point to local cache so model loads from disk after first download
        cache_folder=cache_dir
    )
    # Returns a LangChain-compatible embedding object
    # Can be plugged directly into Chroma, FAISS, or any LangChain vector store