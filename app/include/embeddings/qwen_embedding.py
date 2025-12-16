from langchain.embeddings.base import Embeddings
import dashscope
from app.include.config import config


dashscope.api_key = config.QWEN_API_KEY
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"


class QwenEmbedding(Embeddings):
    def __init__(self, model=config.EMBEDDING_MODEL_ID, dimensions=config.VECTOR_DIMENSION):
        self.model = model
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        resp = dashscope.TextEmbedding.call(
            model=self.model,
            input=[text],
            text_type="query",
            dimensions=self.dimensions
        )
        return resp.output["embeddings"][0]["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = dashscope.TextEmbedding.call(
            model=self.model,
            input=texts,
            text_type="document",
            dimensions=self.dimensions
        )
        return [item["embedding"] for item in response.output["embeddings"]]