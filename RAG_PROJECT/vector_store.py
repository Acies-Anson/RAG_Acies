from langchain_community.vectorstores import FAISS
from typing import List, Any


def create_vector_db(docs: List, embedding_model: Any) -> FAISS:
    """
    Create a FAISS vector database from documents using an embedding model.

    Args:
        docs: List of LangChain Document objects.
        embedding_model: Embedding model used to convert text into vectors.

    Returns:
        FAISS vector store instance.
    """

    # Convert documents into embeddings and store them in FAISS index.
    # distance_strategy='COSINE' means similarity search will use cosine similarity.
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model,
        distance_strategy="COSINE",
    )

    return vector_store
