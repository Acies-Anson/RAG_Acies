def get_top_k_results(vector_db, query, k=5):
    # Performs single-stage semantic retrieval using cosine similarity
    # 
    # Steps happening internally:
    #   1. Query text is converted to a 384-dim vector using the embedding model
    #   2. ChromaDB compares this vector against all stored chunk vectors
    #   3. Cosine similarity score is computed for each chunk
    #   4. Top k chunks ranked by score are returned
    #
    # Returns: list of (Document, score) tuples
    #   - Document contains the chunk text and metadata
    #   - score is cosine similarity between 0.0 and 1.0
    #     (1.0 = identical meaning, 0.0 = completely unrelated)
    #
    # Note: single-stage retrieval — no reranking applied
    # For higher accuracy on complex queries, a cross-encoder
    # reranker should be added as a second stage after this
    return vector_db.similarity_search_with_relevance_scores(query, k=k)