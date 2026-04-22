from langchain_community.vectorstores import FAISS

def create_vector_db(docs, embedding_model):
    return FAISS.from_documents(docs, embedding_model, distance_strategy="COSINE")