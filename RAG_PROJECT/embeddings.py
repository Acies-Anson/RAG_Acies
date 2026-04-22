import os
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    cache_dir = "C:/hf_models"
    os.makedirs(cache_dir, exist_ok=True)
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", 
        cache_folder=cache_dir
    )