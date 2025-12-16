from pathlib import Path
import asyncio
import json
from dotenv import load_dotenv
import os
import traceback
from typing import List
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_qwq import ChatQwQ
from .schemas.dialog import ResponseFormat, ResponseFormatAi, UploadSleepAi
from .RAG.rag_langchain import retrieve_context
from .exceptions import (
    DialogAiErrorConnect,
    DialogAiErrorGeneration,
    DialogAiErrorFormat,
)
from langchain_qdrant import QdrantVectorStore
from app.include.logging_config import logger as log
from app.include.config import config


BASE_DIR = Path(__file__).resolve().parent.parent


async def geration_pipe(sleep_data: UploadSleepAi) -> ResponseFormatAi:
    if not config.QWEN_API_KEY:
        raise DialogAiErrorConnect("API key is not set.")
    
    system_instruction = (BASE_DIR / "context" / "2025-12-12-instruction.txt").read_text()
    help_model_system_instruction = (BASE_DIR / "context" / "2025-11-17-help_model.txt").read_text()

    # Получаем историю диалога пользователя из redis по ключу user_id
    history = []

    
    try:
        agent = create_agent(
            model=ChatQwQ(
                api_key=config.QWEN_API_KEY,
                model=config.MODEL_DIALOG_AI,
                temperature=0.3,
                top_p=0.95,
            ),
            system_prompt=system_instruction,
            response_format=ResponseFormatAi, # поменять
        )
        response = await agent.ainvoke(
            {"messages": 
                [
                    {"role": "user", "content": f"Дневник пользователя: {user_diary_records}"}, 
                    {"role": "user", "content": f"Сегодняшний сон: {sleep_daily_stats}"}, 
                    {"role": "user", "content": f"Сон за последние 3 дня: {sleep_weekly_stats}"}, 
                    {"role": "user", "content": f"История старых советов: {history_sleep_assessment}"},

                    {
                        "role": "system",
                        "content": (
                            "Ниже приведёна 'База знаний Sleeptery'— достоверные научные материалы и объяснения, "
                            "полученные сонологом. "
                            "Ты обязан опираться строго на них при анализе сна и формулировке ответа.\n\n"
                            f"<RAG_CONTEXT>\n{rag_answer}\n</RAG_CONTEXT>\n"
                            "Используй его как главный источник истины при рассуждениях."
                        ),
                    },
                ]
            }, 



        )

        try:
            log.success(f"{response['structured_response']} \n\n")
            # log.success(f"{response.choices[0].message.content} ")
            return response['structured_response']
        except json.JSONDecodeError as e:
            log.warning(f" JSON parsing failed: {e} — retrying...")
            # last_exception = e
            # await asyncio.sleep(1)
            return

    except Exception as e:
        # last_exception = e
        log.warning(f"QWEN generation failed: {e}")

        # await asyncio.sleep(1)


    # log.error(f"All attempts failed: {last_exception}")
    # return


if __name__ == "__main__":
    user_prompt_path = "/sleeptery/Sleeptery-AI/app/sleep_ai/resources/sleep.json"
    with open(user_prompt_path, "r") as f:
        sleep_data = f.read()

    asyncio.run(geration_pipe(sleep_data=json.loads(sleep_data)))
