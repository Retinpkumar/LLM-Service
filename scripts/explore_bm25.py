from llm_service.chunking import chunk_document
from llm_service.retrieval.bm25 import BM25Query
from llm_service.retrieval.fusion import hybrid_search
from llm_service.retrieval.vector_store import QdrantDocumentStore

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
bm25_results = bm25_index.query("patents")

print("=== BM25 ===")
for r in bm25_results:
    print(f"Score: {r['score']:.4f}")
    print(f"Chunk Index: {r['chunk_index']}")
    print(f"Chunk: {r['chunk'][:150]}...")
    print()


store = QdrantDocumentStore()
store.create_collection("coke_10k")
store.store_documents("coke_10k", docs)

vector_results = store.query_documents("coke_10k", "patents")

print("=== VECTOR ===")
for r in vector_results:
    print(f"Score: {r['score']:.4f}")
    print(f"Chunk Index: {r['chunk_index']}")
    print(f"Chunk: {r['chunk'][:150]}...")
    print()

print("=== RRF ===")
fused = hybrid_search("patents", bm25_index, store, "coke_10k")
for idx, score in fused:
    print(f"Chunk Index: {idx} | RRF Score: {score:.4f}")
