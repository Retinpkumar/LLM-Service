def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    fused_scores = {}

    # 1. BM25 list
    for rank, r in enumerate(bm25_results, start=1):
        idx = r["chunk_index"]
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank)

    # 2. Vector list
    for rank, r in enumerate(vector_results, start=1):
        idx = r["chunk_index"]
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank)

    # 3. Sort, highest first
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query, bm25_index, vector_store, collection, top_n=3, k=60):
    bm25_results = bm25_index.query(query, top_n=top_n)
    vector_results = vector_store.query_documents(collection, query, limit=top_n)
    return reciprocal_rank_fusion(bm25_results, vector_results, k=k)
