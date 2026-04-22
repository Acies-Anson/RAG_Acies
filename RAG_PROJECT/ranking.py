def get_top_k_results(vector_db, query, k=5):
    return vector_db.similarity_search_with_relevance_scores(query, k=k)