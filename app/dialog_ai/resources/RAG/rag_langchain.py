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
        test_collection_name = f"{config.COLLECTION_NAME_DIALOG_AI}_test"
        return QdrantVectorStore(
            client=QdrantClient(host="localhost", port=6445),
            collection_name=test_collection_name,
            embedding=embeddings,
            retrieval_mode="dense",
            content_payload_key="content",
            metadata_payload_key="payload"
        )
    else:
        return QdrantVectorStore(
            client=client,
            collection_name=config.COLLECTION_NAME_DIALOG_AI,
            embedding=embeddings,
            retrieval_mode="dense",
            content_payload_key="content",
            metadata_payload_key="payload"
        )

async def retriever_context(
        is_test: bool = False,
    ):
    vector_store = get_vector_store(is_test)
    retriever = vector_store.as_retriever()
    return retriever