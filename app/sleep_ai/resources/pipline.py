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
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_qwq import ChatQwQ

from .schemas.sleepai import ResponseFormat, ResponseFormatAi, UploadSleepAi
from .RAG.rag_langchain import retrieve_context
from .exceptions import (
    SleepAiErrorGeneration,
    SleepAiErrorFormat, 
    SleepAiErrorConnect
)
from app.include.logging_config import logger as log
from app.include.config import config


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
    if not config.QWEN_API_KEY:
        raise SleepAiErrorConnect("API key is not set.")
    
    system_instruction = (BASE_DIR / "context" / "2025-11-12-instruction.txt").read_text()
    help_model_system_instruction = (BASE_DIR / "context" / "2025-11-17-help_model.txt").read_text()

    user_diary_records = extract_full_block(sleep_data["user_diary_records"])
    sleep_daily_stats = extract_full_block(sleep_data["sleep_daily_stats"])
    sleep_weekly_stats = {
        date: extract_full_block(day)
        for date, day in sleep_data["sleep_weekly_stats"].items()
    }
    history_sleep_assessment = [
        extract_full_block(day)
        for day in sleep_data["history_sleep_assessment"]
    ]
    
    agent_helper = create_agent(
        model=ChatQwQ(
            api_key=config.QWEN_API_KEY,
            model="qwen-flash",
            temperature=0.05,
            extra_body={
                "enable_thinking": True,
                "thinking_budget": 120,
            },
        ),
        # model=ChatDeepSeek(
        #     api_key=config.DEEPSEEK_API_KEY,
        #     model="deepseek-chat",
        #     temperature=0,
        #     max_retries=4,
        # ),
        # model=ChatGoogleGenerativeAI(
        #     google_api_key=config.GEMINI_API_KEY,
        #     model="gemini-2.5-flash",
        #     temperature=0.1,
        # ),
        # model=ChatOpenAI(
        #     api_key=config.OPENAI_API_KEY,
        #     model="gpt-5-nano",
        #     max_retries=4,
        #     temperature=0
        # ),
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

    rag_answer = await retrieve_context(topics=problems, is_test=config.TEST_MODE_DB)
    log.info(f"{rag_answer}")

    # last_exception = None
    # for attempt in range(3):
    try:
        agent = create_agent(
            model=ChatQwQ(
                api_key=config.QWEN_API_KEY,
                model=config.MODEL_SLEEP_AI,
                temperature=0.12,
                top_p=0.95,
                extra_body={
                    "enable_thinking": True,
                    "thinking_budget": 600,
                },
            ),
            system_prompt=system_instruction,
            response_format=ResponseFormatAi,
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
        # client = OpenAI(
        #     api_key=config.QWEN_API_KEY,
        #     base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        # )

        # response = client.chat.completions.create(
        #     model="qwen-plus",
        #     top_p=0.95,
        #     temperature=0.1,
        #     extra_body={
        #         "enable_thinking": True,
        #         "thinking_budget": 1000,
        #     },
        #     messages=[
        #         {"role": "system", "content": f"{system_instruction}"},

        #         {"role": "user", "content": f"Дневник пользователя: {user_diary_records}"}, 
        #         {"role": "user", "content": f"Сегодняшний сон: {sleep_daily_stats}"}, 
        #         {"role": "user", "content": f"Сон за последние 3 дня: {sleep_weekly_stats}"}, 
        #         # {"role": "user", "content": f"История советов: {history_sleep_assessment}"},

        #         {
        #             "role": "system",
        #             "content": (
        #                 "Ниже приведёна 'База знаний Sleeptery'— достоверные научные материалы и объяснения, "
        #                 "полученные сонологом. "
        #                 "Ты обязан опираться строго на них при анализе сна и формулировке ответа.\n\n"
        #                 f"<RAG_CONTEXT>\n{rag_answer}\n</RAG_CONTEXT>\n"
        #                 "Используй его как главный источник истины при рассуждениях."
        #             ),
        #         },
        #     ],                
        # )






        # print(resp.choices[0].message.content)

            # model=ChatGoogleGenerativeAI(
            #     google_api_key=config.GEMINI_API_KEY,
            #     # model="gemini-3-pro-preview",
            #     model="gemini-2.5-flash",
            #     temperature=0.1,
            #     max_output_tokens=5000,
            #     top_p=0.95,
            #     thinking_budget=-1, #-1 dynamic
            #     max_retries=4,
            #     safety_settings={
            #         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            #         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            #         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            #         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            #     }

            # ),
            # model=ChatOpenAI(
            #     api_key=config.OPENAI_API_KEY,
            #     model="gpt-5-nano",
            #     temperature=0.1,
            #     top_p=0.95,
            #     max_output_tokens=3000,
            #     max_retries=4,
            # ),

            # system_prompt=system_instruction,
            # response_format=ResponseFormatAi
        # )

        # response = await agent.ainvoke(
        #     {"messages": [
        #         {"role": "user", "content": f"Дневник пользователя: {user_diary_records}"}, 
        #         {"role": "user", "content": f"Сегодняшний сон: {sleep_daily_stats}"}, 
        #         {"role": "user", "content": f"Сон за последние 3 дня: {sleep_weekly_stats}"}, 
        #         # {"role": "user", "content": f"История советов: {history_sleep_assessment}"},

        #         {
        #             "role": "system",
        #             "content": (
        #                 "Ниже приведёна 'База знаний Sleeptery'— аналитические материалы и объяснения, "
        #                 "полученные сонологом. "
        #                 "Ты обязан опираться на них при анализе сна и формулировке ответа.\n\n"
        #                 f"<RAG_CONTEXT>\n{rag_answer}\n</RAG_CONTEXT>\n"
        #                 "Используй его как главный источник истины при рассуждениях."
        #             ),
        #         },
                    
        #     ]},
        # )

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
