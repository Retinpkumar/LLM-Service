import re

import numpy as np
from rank_bm25 import BM25Okapi


def simple_tokenize(doctext):
    return re.findall(r"\b\w+\b", doctext.lower())


class BM25Query:
    def __init__(self, docs):
        self.tokenized_chunks = [simple_tokenize(doc.page_content) for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        self.docs = docs

    def query(self, query, top_n=3):
        tokenized_query = simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            results.append(
                {
                    "score": scores[idx],
                    "chunk_index": self.docs[idx].metadata["chunk_index"],
                    "chunk": self.docs[idx].page_content,
                }
            )
        return results
