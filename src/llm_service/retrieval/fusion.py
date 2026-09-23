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
