import asyncio
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
import time

GEMINI_API_KEY = "AIzaSyBTZrHzTujPbZnnOpMdQDUb9jP0IcyFtx0" 

client = QdrantClient(host="qdrant", port=6333)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY,
    task_type="retrieval_query"
)

def get_vector_store():
    return QdrantVectorStore(
        client=client,
        collection_name="sleep_ai_knowledge_base",
        embedding=embeddings,
        retrieval_mode="dense",
        content_payload_key="text"
    )

async def retrieve_context(topics: list[str]):
    vector_store = get_vector_store()
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
    results = asyncio.run(retrieve_context(query))

    for res in results:
        print(res)
    b = time.time()
    print(f"{b-a}")