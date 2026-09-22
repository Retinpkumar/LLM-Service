from langchain_text_splitters import RecursiveCharacterTextSplitter

# Everythin in a single function named chunk_document(text, headers)
def chunk_document(text, headers, overlap=0):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=overlap,
        add_start_index=True,
    )

    docs = splitter.create_documents([text])

    positions = []
    for header in headers:
        pos = text.find(f"{header}\n")
        positions.append(pos)

    header_positions = list(zip(positions, headers))

    def get_section_for_chunk(start_index, header_positions):
        current_section = None
        for pos, header in header_positions:
            if pos <= start_index:
                current_section = header
            else:
                break
        return current_section

    for i, doc in enumerate(docs):
        section = get_section_for_chunk(doc.metadata["start_index"], header_positions)
        doc.metadata["section"] = section
        doc.metadata["source"] = "coke_10k_excerpt.txt"
        doc.metadata["chunk_index"] = i

    return docs

# Read data/coke_10k_excerpt.txt and find the length
with open("data/coke_10k_excerpt.txt", "r") as f:
    text = f.read()
    length = len(text)
    print(f"Length of data/coke_10k_excerpt.txt: {length} characters")

headers = ['FORWARD-LOOKING STATEMENTS', 'Part I', 'ITEM 1. BUSINESS', 'General', 'Operating Segments', 'Products and Brands', 'Distribution System', 'Promotional and Marketing Programs', 'Investments in Bottling Operations', 'Seasonality', 'Competition', 'Raw Materials', 'Patents, Copyrights, Trade Secrets and Trademarks', 'Governmental Regulation', 'Human Capital Management', 'Available Information', 'ITEM 1A. RISK FACTORS']


# docs1 = chunk_document(text, headers, overlap=0)
# # print docs1[8].metadata and docs1[8].page_content
# print("With overlap=0")
# print(docs1[8].metadata)
# print(docs1[8].page_content)
# print("Length of docs1 with overlap=0: ", len(docs1))

# docs2 = chunk_document(text, headers, overlap=150)
# print("With overlap=150")
# print(docs2[8].metadata)
# print(docs2[8].page_content)
# print("Length of docs2 with overlap=150: ", len(docs2))

from scratch_qdrant_test import QdrantDocumentStore

store = QdrantDocumentStore()
collection_1 = "coke_overlap_0"
collection_2 = "coke_overlap_150"

store.create_collection(collection_1)
docs1 = chunk_document(text, headers, overlap=0)
store.store_documents(collection_1, [doc.page_content for doc in docs1])

store.create_collection(collection_2)
docs2 = chunk_document(text, headers, overlap=150)
store.store_documents(collection_2, [doc.page_content for doc in docs2])

# # Confirm collection sizes
# print(f"Collection '{collection_1}' size: {len(docs1)} documents")
# print(f"Collection '{collection_2}' size: {len(docs2)} documents")

query = "What are Coca-Cola's core values or mission pillars?"

# Query both collections and print results
for collection in [collection_1, collection_2]:
    results = store.query_documents(collection, query)
    print(f"\nResults from collection '{collection}':")
    for score, text in results:
        print(f"Score: {score:.4f}, Text: {text[:100]}...")  # Print first 100 characters of the text


def flag_problem_chunks(docs):
    flagged = []
    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        # logic to find if the chunk is problematic. For example, if the chunk ends without a period or punctuation or starts with a lowercase letter or number or symbols.
        if not text.endswith('.') and not text.endswith('!') and not text.endswith('?') or text[0].islower() or text[0].isdigit() or not text[0].isalpha():
            flagged.append((i, text))
    return flagged

# Flag problematic chunks in both collections
flagged_docs1 = flag_problem_chunks(docs1)
flagged_docs2 = flag_problem_chunks(docs2)

# Print flagged chunks count
print(f"\nFlagged chunks in collection '{collection_1}': {len(flagged_docs1)}")
print(f"Flagged chunks in collection '{collection_2}': {len(flagged_docs2)}")

# Print 3 flagged chunks for collection 1
print(f"\nSample flagged chunks from collection '{collection_1}':")
for i, (index, text) in enumerate(flagged_docs1[:3]):
    print(f"Chunk index: {index}, Text: {text}")

# Print 3 flagged chunks for collection 2
print(f"\nSample flagged chunks from collection '{collection_2}':")
for i, (index, text) in enumerate(flagged_docs2[:3]):   
    print(f"Chunk index: {index}, Text: {text}")

