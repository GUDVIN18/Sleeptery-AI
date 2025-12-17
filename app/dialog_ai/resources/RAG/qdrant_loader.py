from pathlib import Path
from app.include.logging_config import logger as log
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.include.config import config
from app.include.embeddings.qwen_embedding import QwenEmbedding
import uuid
from .create_md_2 import process_book_with_ai


embeddings_qwen = QwenEmbedding(
    model=config.EMBEDDING_MODEL_ID,
    dimensions=config.VECTOR_DIMENSION
)

qdrant_client = QdrantClient(host="localhost", port=6445)

class SleepAiRagEmbeddingConfig:
    @staticmethod
    def run_pipeline(file_paths: list[Path]):
        # Пересоздаем коллекцию для чистоты теста
        # if qdrant_client.collection_exists(collection_name=f"{config.COLLECTION_NAME_DIALOG_AI}_test"):
        #     qdrant_client.delete_collection(collection_name=f"{config.COLLECTION_NAME_DIALOG_AI}_test")
            
        # log.info(f"Создание коллекции: {f'{config.COLLECTION_NAME_DIALOG_AI}_test'}")
        # qdrant_client.recreate_collection(
        #     collection_name=f"{config.COLLECTION_NAME_DIALOG_AI}_test",
        #     vectors_config=VectorParams(size=config.VECTOR_DIMENSION, distance=Distance.COSINE)
        # )

        for file in file_paths:
            log.info(f"\n📘 Обработка файла: {file.name}")
            docs_processed = process_book_with_ai(file)
            
            # Готовим тексты для эмбеддинга
            batch_texts = [d['vector_text'] for d in docs_processed]
            
            # Получаем вектора
            try:
                vectors = SleepAiRagEmbeddingConfig.get_batch_embeddings(batch_texts)
            except Exception as e:
                log.error(f"Critical Error during embedding: {e}")
                return

            points = []
            for i, doc in enumerate(docs_processed):
                if i >= len(vectors): 
                    break
                    
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()), # Генерируем уникальный ID
                        vector=vectors[i],
                        payload={
                            "original_file": file.name,
                            "chapter": doc['payload'].get('chapter', 'Общее'),
                            "subtitle": doc['payload'].get('subchapter', 'Нет подглавы'),
                            "topic": doc['payload'].get('topic', 'Нет темы'),
                            "section": doc['payload'].get('section', 'Нет секции'),
                            "content": doc['payload'].get('content', 'Пустой текст'),
                            "full_context": doc['vector_text']
                        }
                    )
                )
            
            log.info(f"Загрузка {len(points)} точек в Qdrant...")
            for batch_start in tqdm(range(0, len(points), config.BATCH_SIZE)):
                batch_points = points[batch_start:batch_start + config.BATCH_SIZE]
                qdrant_client.upsert(
                    collection_name=f"{config.COLLECTION_NAME_DIALOG_AI}_test",
                    points=batch_points
                )

        log.info("\n✅ Загрузка завершена!")

    @staticmethod
    def get_batch_embeddings(texts: list) -> list:
        all_embeddings = []
        safe_batch_size = 5 
        for i in tqdm(range(0, len(texts), safe_batch_size), desc="Получение эмбеддингов"):
            batch = texts[i:i + safe_batch_size]
            embeddings = embeddings_qwen.embed_documents(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

if __name__ == "__main__":
    SleepAiRagEmbeddingConfig.run_pipeline(
        file_paths=[
            # Path("app/dialog_ai/resources/RAG/knowledge_base/book_1-56.pdf"),
            Path("app/dialog_ai/resources/RAG/knowledge_base/book_56-282pdf.pdf")
        ] 
    )