from llm_service.chunking import chunk_document
from llm_service.retrieval.bm25 import BM25Query

with open("data/coke_10k_excerpt.txt", "r") as f:
    text = f.read()

headers = [
    "FORWARD-LOOKING STATEMENTS",
    "Part I",
    "ITEM 1. BUSINESS",
    "General",
    "Operating Segments",
    "Products and Brands",
    "Distribution System",
    "Promotional and Marketing Programs",
    "Investments in Bottling Operations",
    "Seasonality",
    "Competition",
    "Raw Materials",
    "Patents, Copyrights, Trade Secrets and Trademarks",
    "Governmental Regulation",
    "Human Capital Management",
    "Available Information",
    "ITEM 1A. RISK FACTORS",
]

docs = chunk_document(text, headers, source_file="coke_10k_excerpt.txt", overlap=150)

bm25_index = BM25Query(docs)
results = bm25_index.query("patents")

for r in results:
    print(f"Score: {r['score']:.4f}")
    print(f"Chunk Index: {r['chunk_index']}")
    print(f"Chunk: {r['chunk'][:150]}...")
    print()
