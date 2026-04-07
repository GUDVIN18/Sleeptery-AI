import os
import torch
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
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, CrossEncoderReranker
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_transformers import LongContextReorder
from app.include.logging_config import logger as log
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from flashrank import Ranker


embeddings = QwenEmbedding(
    model=config.EMBEDDING_MODEL_ID,
    dimensions=config.VECTOR_DIMENSION
)

try:
    _cross_encoder = HuggingFaceCrossEncoder(
        model_name="DiTy/cross-encoder-russian-msmarco",
        model_kwargs={"trust_remote_code": True},
    )
    reranker = CrossEncoderReranker(model=_cross_encoder, top_n=12)
    log.info("GTE multilingual reranker loaded successfully")
except Exception as e:
    log.error(f"Failed to load reranker: {e}")
    reranker = None


def get_vector_store(is_test):
    if is_test:
        test_collection_name = f"{config.COLLECTION_NAME_DIALOG_AI}"
        return QdrantVectorStore(
            client=QdrantClient(host="82.22.184.82", port=6333),
            collection_name=test_collection_name,
            embedding=embeddings,
            retrieval_mode="dense",
            content_payload_key="content",
            metadata_payload_key="payload"
        )
    else:
        return QdrantVectorStore(
            client=QdrantClient(host="qdrant", port=6333),
            collection_name=config.COLLECTION_NAME_DIALOG_AI,
            embedding=embeddings,
            retrieval_mode="dense",
            content_payload_key="content",
            metadata_payload_key="payload"
        )


async def retriever_context(is_test: bool = False) -> ContextualCompressionRetriever:
    vector_store = get_vector_store(is_test)

    try:
        # Базовый ретривер
        base_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 20, "lambda_mult": 0.7}
        )

        if reranker is None:
            log.warning("Reranker unavailable, fallback to MMR k=5")
            return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

        pipeline_compressor = DocumentCompressorPipeline(transformers=[reranker])
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