from qdrant_client import QdrantClient

client = QdrantClient(host="qdrant", port=6333)

points = client.scroll(
    collection_name="sleep_ai_knowledge_base",
    limit=5,
    with_payload=True,
    with_vectors=False
)

print(points)