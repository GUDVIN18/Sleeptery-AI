from pathlib import Path
import pdfplumber
import os
from app.include.logging_config import logger as log
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from app.include.config import config
from app.include.embeddings.qwen_embedding import QwenEmbedding

# embeddings_gemini = GoogleGenerativeAIEmbeddings(
#     model=f"models/{config.EMBEDDING_MODEL_ID}",
#     google_api_key=config.GEMINI_API_KEY,
#     task_type="retrieval_document"
# )


embeddings_qwen = QwenEmbedding(
    model=config.EMBEDDING_MODEL_ID,
    dimensions=config.VECTOR_DIMENSION
)

client = OpenAI(
    api_key=config.QWEN_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
qdrant_client=QdrantClient(host="localhost", port=6445)
text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ('##', 'advice'), 
        ('###', 'longrid')
    ]
)


class SleepAiRagEmbeddingConfig:
    @staticmethod
    def run_pipeline(file_paths: list[Path]):
        if not qdrant_client.collection_exists(collection_name=f"{config.COLLECTION_NAME_SLEEP_AI}_test"):
            log.info(f"Создание коллекции: {f"{config.COLLECTION_NAME_SLEEP_AI}_test"}")
            qdrant_client.recreate_collection(
                collection_name=f"{config.COLLECTION_NAME_SLEEP_AI}_test",
                vectors_config=VectorParams(size=config.VECTOR_DIMENSION, distance=Distance.COSINE)
            )
        else:
            log.info(f"Коллекция {f"{config.COLLECTION_NAME_SLEEP_AI}_test"} уже существует.")

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
                    collection_name=f"{config.COLLECTION_NAME_SLEEP_AI}_test",
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
            for batch_start_index in tqdm(range(0, len(all_points), config.BATCH_SIZE), desc="Qdrant upload"):
                batch_points = all_points[batch_start_index:batch_start_index + config.BATCH_SIZE]
                qdrant_client.upsert(
                    collection_name=f"{config.COLLECTION_NAME_SLEEP_AI}_test",
                    points=batch_points,
                    wait=True
                )

        log.info("\n✅ Загрузка всех файлов завершена!")
        info = qdrant_client.get_collection(collection_name=f"{config.COLLECTION_NAME_SLEEP_AI}_test")
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
        for i in tqdm(range(0, len(merged_list), config.BATCH_SIZE), desc="Получение эмбеддингов"):
            batch = merged_list[i:i + config.BATCH_SIZE]
            for item in batch:
                combined_text = f"{item['advice']}\n{item['advice_text']}\n\n{item['longrid']}\n{item['longrid_text']}".strip()
                # response = embeddings_gemini.embed_documents(
                #     texts=[combined_text],
                #     titles=[item["advice"]]
                # )
                # all_embeddings.append(response[0])
                response = embeddings_qwen.embed_documents(
                    texts=[combined_text],
                )
                all_embeddings.append(response[0])
        return all_embeddings




if __name__ == "__main__":
    # qdrant_client.delete_collection(collection_name=f"{config.COLLECTION_NAME_SLEEP_AI}_test")
    SleepAiRagEmbeddingConfig.run_pipeline(
        file_paths=[
            Path("app/sleep_ai/resources/RAG/knowledge_base/Сон.md"),
        ]
    )

