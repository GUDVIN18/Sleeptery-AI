from openai import OpenAI
from pathlib import Path
import asyncio
import json
from dotenv import load_dotenv
import os
import traceback
from typing import List
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

from .schemas.sleepai import ResponseFormat, ResponseFormatAi, UploadSleepAi
from .RAG.rag_langchain import retrieve_context
from .exceptions import (
    SleepAiErrorGeneration,
    SleepAiErrorFormat, 
    SleepAiErrorConnect
)
from app.include.logging_config import logger as log
from include.config import config


BASE_DIR = Path(__file__).resolve().parent.parent

def extract_full_block(block: dict) -> dict:
    result = {}
    for key, value in block.items():
        if value is None:
            result[key] = {
                "amount": None,
                "type": None,
                "description": None
            }
            continue

        result[key] = {
            "amount": value.get("amount"),
            "type": value.get("type"),
            "description": value.get("description")
        }
    return result

async def geration_pipe(sleep_data: UploadSleepAi) -> ResponseFormatAi:
    if not config.DEEPSEEK_API_KEY:
        raise SleepAiErrorConnect("API key is not set.")
    
    system_instruction = (BASE_DIR / "context" / "2025-11-12-instruction.txt").read_text()
    help_model_system_instruction = (BASE_DIR / "context" / "2025-11-17-help_model.txt").read_text()

    user_diary_records = extract_full_block(sleep_data["user diary records"])
    sleep_daily_stats = extract_full_block(sleep_data["sleep daily stats"])
    sleep_weekly_stats = {
        date: extract_full_block(day)
        for date, day in sleep_data["sleep weekly stats"].items()
    }
    log.info(f"{user_diary_records=}\n\n{sleep_daily_stats=}\n\n{sleep_weekly_stats=}")
    # history_sleep_assessment = sleep_data
    
    agent_helper = create_agent(
        model=ChatDeepSeek(
            api_key=config.DEEPSEEK_API_KEY,
            model=config.MODEL,
            temperature=0.1,
        ),
        system_prompt=help_model_system_instruction,
        response_format=ResponseFormat
    )

    helper_analytics = await agent_helper.ainvoke(
        {"messages": 
            [
                {"role": "user", "content": f"Данные сна: {sleep_data}"},
            ]
        }
    )
    problems = helper_analytics['structured_response'].analysis
    log.info(f"Extracted problems: {problems} ")

    rag_answer = await retrieve_context(topics=problems)
    print(rag_answer)

    last_exception = None
    for attempt in range(3):
        try:
            agent = create_agent(
                model=ChatDeepSeek(
                    api_key=config.DEEPSEEK_API_KEY,
                    model="deepseek-chat",
                    temperature=0.14,
                    max_tokens=1000,
                    top_p=0.95,
                    frequency_penalty=0.8,
                    presence_penalty=0.6,
                    # timeout=10, 
                ),
                system_prompt=system_instruction,
                response_format=ResponseFormatAi
            )

            response = await agent.ainvoke(
                {"messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ниже приведёна 'База знаний Sleeptery'— аналитические материалы и объяснения, "
                            "полученные сонологом. "
                            "Ты обязан опираться на них при анализе сна и формулировке ответа.\n\n"
                            f"<RAG_CONTEXT>\n{rag_answer}\n</RAG_CONTEXT>\n"
                            "Используй его как главный источник истины при рассуждениях."
                        ),
                    },
                    {"role": "user", "content": f"Дневник пользователя: {user_diary_records}"}, 
                    {"role": "user", "content": f"Сегодняшний сон: {sleep_daily_stats}"}, 
                    {"role": "user", "content": f"Сон за последние 3 дня: {sleep_weekly_stats}"}, 
                    # {"role": "user", "content": f"История советов: {history_sleep_assessment}"},
                      
                ]},
            )

            try:
                log.success(f"{response['structured_response']} ")
                return response['structured_response']
            except json.JSONDecodeError as e:
                log.warning(f"[Attempt {attempt+1}/3] JSON parsing failed: {e} — retrying...")
                last_exception = e
                await asyncio.sleep(1)

        except Exception as e:
            last_exception = e
            log.warning(f"[Attempt {attempt+1}/3] DeepSeek generation failed: {e}")
    
            await asyncio.sleep(1)

    log.error(f"All attempts failed: {last_exception}")
    return


if __name__ == "__main__":
    user_prompt_path = "/Applications/xdev/sleep_ai_rag/app/ai/resources/sleep.json"
    with open(user_prompt_path, "r") as f:
        sleep_data = f.read()
    asyncio.run(geration_pipe(sleep_data=sleep_data))