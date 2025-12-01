from google import genai
from google.genai import types
from pathlib import Path
import pdfplumber
import os
from dotenv import load_dotenv
from loguru import logger as log
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter


load_dotenv()

EMBEDDING_MODEL_ID = "text-embedding-004"  # text-embedding-004 / gemini-embedding-001

QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
COLLECTION_NAME = "sleep_ai_knowledge_base"
VECTOR_DIMENSION = 768  # 768 для text-embedding-004, 3072 для gemini-embedding-001
BATCH_SIZE = 150

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ('##', 'advice'), 
        ('###', 'longrid')
    ]
)


class SleepAiRagEmbeddingConfig:
    @staticmethod
    def run_pipeline(file_paths: list[Path]):
        if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            log.info(f"Создание коллекции: {COLLECTION_NAME}")
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE)
            )
        else:
            log.info(f"Коллекция {COLLECTION_NAME} уже существует.")

        # общий счётчик для всех файлов
        global_id = 0

        for file in file_paths:
            log.info(f"\n📘 Обработка файла: {file.name}")
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            chunks_data, merged_list = SleepAiRagEmbeddingConfig.create_chunks_data(content=content)

            embeddings = SleepAiRagEmbeddingConfig.get_batch_embeddings(merged_list=merged_list)

            all_points = []
            for num, data in enumerate(chunks_data):
                # Проверяем, существует ли точка уже в Qdrant
                exists = qdrant_client.count(
                    collection_name=COLLECTION_NAME,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="advice",
                                match=models.MatchValue(value=data["advice"])
                            ),
                            models.FieldCondition(
                                key="chunk_id",
                                match=models.MatchValue(value=data["chunk_id"])
                            ),
                            models.FieldCondition(
                                key="source_file",
                                match=models.MatchValue(value=file.name)
                            )
                        ]
                    )
                )

                if exists.count > 0:
                    log.info(f"⏩ Пропуск: advice='{data['advice']}', chunk_id={data['chunk_id']} — уже существует")
                    continue

                all_points.append(
                    models.PointStruct(
                        id=global_id + num,
                        vector=embeddings[num],
                        payload={
                            "title_doc": data["title_doc"], # заголовок документа
                            "advice": data["advice"],     
                            "text": data["text"],
                            "chunk_id": data["chunk_id"],
                            "source_file": file.name
                        }
                    )
                )
            global_id += len(chunks_data)

            log.info(f"Загрузка {len(all_points)} точек в Qdrant...")
            for batch_start_index in tqdm(range(0, len(all_points), BATCH_SIZE), desc="Qdrant upload"):
                batch_points = all_points[batch_start_index:batch_start_index + BATCH_SIZE]
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch_points,
                    wait=True
                )

        log.info("\n✅ Загрузка всех файлов завершена!")
        info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        log.info(f"Текущее количество точек: {info.points_count}")



    @staticmethod
    def create_chunks_data(content: str) -> list:
        chanks = text_splitter.split_text(content)
        merged = {}

        for doc in chanks:
            advice = doc.metadata.get("advice")
            longrid = doc.metadata.get("longrid")
            text = doc.page_content.strip()
            if advice not in merged:
                merged[advice] = {
                    "advice": advice,
                    "advice_text": text if not longrid else "",
                    "longrid": longrid,
                    "longrid_text": text if longrid else "",
                    "chunk_id": len(merged)
                }

            else:
                if longrid:
                    merged[advice]["longrid"] = longrid
                    merged[advice]["longrid_text"] = text
                else:
                    merged[advice]["advice_text"] += f"\n\n{text}"

        merged_list = list(merged.values())
        log.info(f"Объединено {len(merged_list)} советов.")

        chunks_data = []
        for item in merged_list:
            chunks_data.append({
                "title_doc": "Как улучишить сон",
                "advice": item["advice"],
                "text": f"{item['advice']}\n{item['advice_text']}\n\n{item['longrid_text']}".strip(),
                "chunk_id": item["chunk_id"]
            })
        return chunks_data, merged_list


    @staticmethod
    def get_batch_embeddings(merged_list: list) -> list:
        all_embeddings = []
        for i in tqdm(range(0, len(merged_list), BATCH_SIZE), desc="Получение эмбеддингов"):
            batch = merged_list[i:i + BATCH_SIZE]
            for item in batch:
                combined_text = f"{item['advice']}\n{item['advice_text']}\n\n{item['longrid']}\n{item['longrid_text']}".strip()
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL_ID,
                    contents=[combined_text],
                    config=types.EmbedContentConfig(
                        task_type="retrieval_document",
                        title=item["advice"]
                    )
                )
                all_embeddings.append(response.embeddings[0].values)
        return all_embeddings




if __name__ == "__main__":
    # qdrant_client.delete_collection(collection_name="sleep_ai_knowledge_base")
    SleepAiRagEmbeddingConfig.run_pipeline(
        file_paths=[
            Path("app/ai/resources/RAG/knowledge_base/Сон.md"),
        ]
    )

