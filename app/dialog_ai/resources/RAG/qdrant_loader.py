from pathlib import Path
from app.include.logging_config import logger as log
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.include.config import config
from app.include.embeddings.qwen_embedding import QwenEmbedding
import uuid


COLLECTION_NAME = "sleepteryGPT"

embeddings_qwen = QwenEmbedding(
    model=config.EMBEDDING_MODEL_ID,
    dimensions=config.VECTOR_DIMENSION
)

qdrant_client = QdrantClient(host="localhost", port=6445)

# 1. ГРУБАЯ РАЗБИВКА (По главам)
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ('#', 'chapter'), 
        ('##', 'subtitle'),
    ],
    strip_headers=False 
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,       
    chunk_overlap=50,
    length_function=len,
    separators=["\n", ". ", "! ",]
)

class SleepAiRagEmbeddingConfig:
    @staticmethod
    def run_pipeline(file_paths: list[Path]):
        # Пересоздаем коллекцию для чистоты теста
        if qdrant_client.collection_exists(collection_name=f"{COLLECTION_NAME}_test"):
            qdrant_client.delete_collection(collection_name=f"{COLLECTION_NAME}_test")
            
        log.info(f"Создание коллекции: {f'{COLLECTION_NAME}_test'}")
        qdrant_client.recreate_collection(
            collection_name=f"{COLLECTION_NAME}_test",
            vectors_config=VectorParams(size=config.VECTOR_DIMENSION, distance=Distance.COSINE)
        )

        for file in file_paths:
            log.info(f"\n📘 Обработка файла: {file.name}")
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Разбиваем красиво
            docs_processed = SleepAiRagEmbeddingConfig.process_markdown(content, file.name)
            
            # Готовим тексты для эмбеддинга
            batch_texts = [d['page_content_for_embedding'] for d in docs_processed]
            
            # Получаем вектора
            try:
                vectors = SleepAiRagEmbeddingConfig.get_batch_embeddings(batch_texts)
            except Exception as e:
                log.error(f"Critical Error during embedding: {e}")
                return

            points = []
            for i, doc in enumerate(docs_processed):
                # Если векторов вернулось меньше чем текстов (сбой), пропускаем
                if i >= len(vectors): 
                    break
                    
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()), # Генерируем уникальный ID
                        vector=vectors[i],
                        payload={
                            "source_file": file.name,
                            "chapter": doc['metadata'].get('chapter', 'Общее'),
                            "subtitle": doc['metadata'].get('subtitle', ''),
                            "text": doc['text_content'], # Чистый текст для показа юзеру
                            "full_context": doc['page_content_for_embedding']
                        }
                    )
                )
            
            log.info(f"Загрузка {len(points)} точек в Qdrant...")
            for batch_start in tqdm(range(0, len(points), config.BATCH_SIZE)):
                batch_points = points[batch_start:batch_start + config.BATCH_SIZE]
                qdrant_client.upsert(
                    collection_name=f"{COLLECTION_NAME}_test",
                    points=batch_points
                )

        log.info("\n✅ Загрузка завершена!")

    @staticmethod
    def process_markdown(content: str, filename: str) -> list:
        final_chunks = []
        md_docs = markdown_splitter.split_text(content)
        split_docs = text_splitter.split_documents(md_docs)
        
        for doc in split_docs:
            meta = doc.metadata
            text = doc.page_content.strip()
            
            if len(text) < 20: # Пропускаем мусор и слишком короткие заголовки без текста
                continue

            chapter = meta.get('chapter', '')
            subtitle = meta.get('subtitle', '')
            
            # Формируем контекст для ИИ (чтобы он понимал о чем речь)
            # Но сам text сохраняем чистым
            content_for_embedding = f"Тема: {chapter} -> {subtitle}\nТекст: {text}"
            
            final_chunks.append({
                "text_content": text,
                "page_content_for_embedding": content_for_embedding,
                "metadata": meta
            })
            
        log.info(f"Разбито на {len(final_chunks)} аккуратных чанков.")
        # Для отладки покажем пример первых 2 чанков
        if final_chunks:
            log.info(f"Пример чанка #1:\n---\n{final_chunks[0]['text_content']}\n---")
            
        return final_chunks

    @staticmethod
    def get_batch_embeddings(texts: list) -> list:
        all_embeddings = []
        # Батч можно уменьшить, если API падает
        safe_batch_size = 5 
        for i in tqdm(range(0, len(texts), safe_batch_size), desc="Получение эмбеддингов"):
            batch = texts[i:i + safe_batch_size]
            embeddings = embeddings_qwen.embed_documents(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

if __name__ == "__main__":
    SleepAiRagEmbeddingConfig.run_pipeline(
        file_paths=[Path("app/dialog_ai/resources/RAG/knowledge_base/book_test.md")] 
    )