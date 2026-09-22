from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

load_dotenv()

class QdrantDocumentStore:
    def __init__(self, qdrant_url="http://localhost:6333", model_name="all-MiniLM-L6-v2"):
        self.client = QdrantClient(url=qdrant_url)
        self.model = SentenceTransformer(model_name)

    def create_collection(self, collection, vector_size=384, distance=Distance.COSINE):
        if self.client.collection_exists(collection_name=collection):
            print(f"Collection '{collection}' already exists — skipping creation.")
            return
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )

    def store_documents(self, collection, documents):
        vectors = self.model.encode(documents)
        points = [
            PointStruct(id=idx, vector=vector.tolist(), payload={"text": doc})
            for idx, (doc, vector) in enumerate(zip(documents, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points)
        print(f"Inserted {len(points)} points into collection '{collection}'.")

    def query_documents(self, collection, query, limit=3):
        query_vector = self.model.encode(query).tolist()
        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit
        )
        return [(point.score, point.payload["text"]) for point in results.points]