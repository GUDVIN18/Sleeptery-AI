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
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_transformers import LongContextReorder
from app.include.logging_config import logger as log
from flashrank import Ranker


client = QdrantClient(host="qdrant", port=6333)

embeddings = QwenEmbedding(
    model=config.EMBEDDING_MODEL_ID,
    dimensions=config.VECTOR_DIMENSION
)
def get_vector_store(is_test):
    if is_test:
        test_collection_name = f"{config.COLLECTION_NAME_DIALOG_AI}"
        return QdrantVectorStore(
            client=QdrantClient(host="82.22.184.82", port=6445),
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


try:
    flashrank_client = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank_cache")
except Exception as e:
    log.error(f"Failed to load Ranker: {e}")
    flashrank_client = None

async def retriever_context(is_test: bool = False):
    vector_store = get_vector_store(is_test)

    try:
        if flashrank_client is None:
            raise ValueError("Flashrank client is not initialized")

        # Базовый ретривер
        base_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 45, "lambda_mult": 0.7}
        )

        # base_retriever = vector_store.as_retriever(
        #     search_type="similarity",
        #     search_kwargs={
        #         "k": 25,  # Берем больше документов для анализа (было 10)
        #     }
        # )

        reranker = FlashrankRerank(
            client=flashrank_client,
            model=config.FLASH_RANK_MODEL, 
            top_n=10
        )

        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[reranker]
        )

        return ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, 
            base_retriever=base_retriever
        )

    except Exception as e:
        log.error(f"Fallback to MMR due to: {e}")
        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5}) 
    finally:
        import gc
        gc.collect()