from qdrant_client import QdrantClient

client = QdrantClient(host="qdrant", port=6333)

points = client.scroll(
    collection_name="insomnia_book",
    limit=5,
    with_payload=True,
    with_vectors=False
)

print(points)