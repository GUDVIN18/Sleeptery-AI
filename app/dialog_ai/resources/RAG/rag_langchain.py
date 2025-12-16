import os
import asyncio
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
import time
from app.include.config import config
import dashscope
from qdrant_client.models import Distance, VectorParams
from app.include.embeddings.qwen_embedding import QwenEmbedding


client = QdrantClient(host="qdrant", port=6333)

embeddings = QwenEmbedding(
    model=config.EMBEDDING_MODEL_ID,
    dimensions=config.VECTOR_DIMENSION
)
def get_vector_store(is_test):
    if is_test:
        test_collection_name = f"{config.COLLECTION_NAME}_test"
        return QdrantVectorStore(
            client=QdrantClient(host="localhost", port=6445),
            collection_name=test_collection_name,
            embedding=embeddings,
            retrieval_mode="dense",
            content_payload_key="text"
        )
    else:
        return QdrantVectorStore(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding=embeddings,
            retrieval_mode="dense",
            content_payload_key="text"
        )

async def retrieve_context(
        topics: list[str],
        is_test: bool = False,
    ):
    vector_store = get_vector_store(is_test)
    results_list = []
    
    for topic in topics:
        filter_condition = rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="advice", 
                    # match=rest_models.MatchValue(value=topic)
                    match=rest_models.MatchText(text=topic) 
                )
            ]
        )

        docs = await vector_store.asimilarity_search(
            query=topic,
            k=3,
            filter=filter_condition 
        )
        if docs:
            results_list.append(docs)

    cleaned_contexts = []
    
    for docs in results_list:
        if not docs:
            continue # Если ничего не нашлось по фильтру
            
        for doc in docs:
            cleaned_contexts.append({
                "text": doc.page_content, 
            })

    return cleaned_contexts


if __name__ == "__main__":
    a = time.time()
    query = ['Совет 31: Режим сна', 'Совет 4: Сколько нужно спать?'] 
    results = asyncio.run(retrieve_context(query, is_test=True))

    for res in results:
        print(res)
    b = time.time()
    print(f"{b-a}")