import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel, Field
from langchain_qwq import ChatQwQ
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from app.include.logging_config import logger as log
from app.include.config import config
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm


class ResponseFormatAi(BaseModel):
    chapter: str = Field(description="Название главы (из контекста)")
    subchapter: str = Field(description="Название подглавы (из контекста)")
    chunks: List[str] = Field(description="Список логических смысловых чанков. Каждый чанк - законченная мысль.")

system_prompt_text = """
Ты — эксперт по структурированию научной и медицинской литературы.
Твоя задача — взять текст книжного раздела и разбить его на смысловые фрагменты (чанки) для векторной базы данных.

ПРАВИЛА:
1. Поле `chunks`: разбей поле `content` на список строк.
   - Один чанк = одна законченная мысль, абзац или конкретная инструкция.
   - Если в тексте есть список (1., 2., 3.), старайся не разрывать его, либо делай каждый пункт отдельным чанком, если они длинные.
   - Сохраняй терминологию и исходный текст на русском языке.
   - Не добавляй ничего от себя, только структурируй исходный текст.

2. Строго! Не допускай, чтобы данные выражения попали в чанк без пояснения:
    - Как победить бессонницу? Здоровый сон за 6 недель
    - Неделя 2. Методики ограничения сна и контроля стимула
    - Неделя 3. Образ жизни и гигиена сна
    - Неделя 4. Работа с негативными мыслями и убеждениями
    - Неделя 5. Техники релаксации
    - Неделя 6. Гигиена спальни
    - Завершение программы
    - Лечение острой бессонницы
    - Отзывы о программе когнитивно-поведенческой терапии 
"""

llm = ChatQwQ(
    api_key=config.QWEN_API_KEY,
    model="qwen-flash", 
    temperature=0.1,
    extra_body={
        "enable_thinking": True,
        "thinking_budget": 64,
    },
)
agent_helper = create_agent(
    model=llm,
    system_prompt=system_prompt_text,
    response_format=ResponseFormatAi
)

def parse_pdf_structure(file_path: Path):
    with open("app/dialog_ai/resources/RAG/knowledge_base/book_parsed_56-282.md", "r", encoding="utf-8") as f:
        md_text = f.read()


    # md_text = pymupdf4llm.to_markdown(str(file_path))
    
    # garbage_phrases = [
    #     r"Как победить бессонницу\? Здоровый сон за 6 недель", 
    #     r"Как победить бессонницу? Здоровый сон за 6 недель",
    #     r"Отзывы о программе когнитивно-поведенческой терапии",
    #     r"**VIР    STANDART**"
    # ]
    
    # # 1. Сначала удаляем строки, где ЕСТЬ цифра страницы И мусорный текст
    # # Это прибьет строки типа "**64** Как победить бессонницу..."
    # for phrase in garbage_phrases:
    #     pattern = f"(?<!# ){phrase}" 
    #     md_text = re.sub(pattern, "", md_text, flags=re.IGNORECASE)

    # md_text = re.sub(r"^\s*\*\*\d+\*\*\s+(?![#]).*$", "", md_text, flags=re.MULTILINE)

    # # --- ШАГ 3: Чистим номер страницы, если он прилип к заголовку ---
    # # Если строка вида: "**64** # Заголовок" -> превращаем в "# Заголовок"
    # md_text = re.sub(r"^\s*\*\*\d+\*\*\s+(?=#)", "", md_text, flags=re.MULTILINE)

    # # --- ШАГ 4: Удаляем одинокие номера страниц ---
    # md_text = re.sub(r"^\s*\*\*\d+\*\*\s*$", "", md_text, flags=re.MULTILINE)

    # # --- ШАГ 5: Склеиваем переносы слов (бессон-\nница) ---
    # md_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", md_text)

    # # --- ШАГ 6: Убираем лишние энтеры ---
    # md_text = re.sub(r"\n{3,}", "\n\n", md_text)
    

    # # надо убрать Как победить бессонницу? Здоровый сон за 6 недель **93**
    # with open("app/dialog_ai/resources/RAG/knowledge_base/book_parsed.md", "w", encoding="utf-8") as f:
    #     f.write(md_text)

    headers_to_split_on = [
        ("#", "chapter"),
        ("##", "subchapter"),
        ("###", "topic"),
        ("####", "section"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(md_text)    
    return docs

def process_book_with_ai(file_paths: Path):
    # Шаг 1: Получаем "сырые" секции из книги
    raw_docs = parse_pdf_structure(file_paths)
    log.info(f"Книга разбита на {len(raw_docs)} секций по оглавлению.")
    final_dataset = []

    # Шаг 2: Проходим по каждой секции агентом
    for i, doc in enumerate(raw_docs):
        chapter = doc.metadata.get("chapter", "Тема")
        subchapter = doc.metadata.get("subchapter", "Поддтема")
        topic = doc.metadata.get("topic", "")
        section = doc.metadata.get("section", "")
        content = doc.page_content

        if len(content) < 50:
            continue

        log.info(f"Processing [{i+1}/{len(raw_docs)}]: {chapter} -> {subchapter} -> {topic} -> {section}")

        try:
            # Формируем промпт для агента
            user_msg = f"""
            Вот данные для обработки:
            ГЛАВА: {chapter}
            ПОДГЛАВА: {subchapter}
            ТЕМА: {topic}
            СЕКЦИЯ: {section}
            
            ТЕКСТ ДЛЯ РАЗБИВКИ:
            {content}
            """

            result = agent_helper.invoke({"messages": [("user", user_msg)]})
            log.info(result["structured_response"].chunks)
            parsed_data: ResponseFormatAi = result["structured_response"]
            
            # Сохраняем результат
            for chunk in parsed_data.chunks:
                record = {
                    "vector_text": f"Глава: {parsed_data.chapter}. Подтема: {parsed_data.subchapter}. Тема: {topic}. Секция: {section}. Текст: {chunk}",
                    "payload": {
                        "chapter": parsed_data.chapter,
                        "subchapter": parsed_data.subchapter,
                        "topic": topic,
                        "section": section,
                        "content": chunk,
                        "original_file": file_paths.name
                    }
                }
                final_dataset.append(record)

        except Exception as e:
            log.error(f"Ошибка на блоке {chapter}: {e}")
            return 

    return final_dataset

if __name__ == "__main__":
    book_path = Path("app/dialog_ai/resources/RAG/knowledge_base/book_test_56.pdf")
    data = process_book_with_ai(book_path)
    log.info(f"Успешно обработано чанков: {len(data)}")