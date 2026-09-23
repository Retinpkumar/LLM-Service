from rank_bm25 import BM25Okapi

corpus = ["Revenue grew significantly", "revenue declined slightly"]
tokenized = [doc.lower().split(" ") for doc in corpus]
print(tokenized)
