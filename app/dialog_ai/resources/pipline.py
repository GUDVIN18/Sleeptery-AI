from pathlib import Path
import asyncio
import json
import os
import traceback
from typing import List
from redis import asyncio as aredis
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_qwq import ChatQwQ
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import PydanticOutputParser
from .redis_client import RedisClient
from .RAG.rag_langchain import retriever_context
from .schemas.dialog import ResponseFormatAi, UploadDialogAi
from .exceptions import (
    DialogAiErrorConnect,
    DialogAiErrorGeneration,
    DialogAiErrorFormat,
)
from langchain_qdrant import QdrantVectorStore
from app.include.logging_config import logger as log
from app.include.config import config

    
BASE_DIR = Path(__file__).resolve().parent.parent

async def geration_pipe(data: UploadDialogAi) -> ResponseFormatAi:
    if not config.QWEN_API_KEY:
        raise DialogAiErrorConnect("API key is not set.")
    
    system_instruction = (BASE_DIR / "context" / "2025-12-12-instruction.txt").read_text()
    contextualize_q_system_prompt = (BASE_DIR / "context" / "2025-12-16-contextualize_prompt.txt").read_text()
    # help_model_system_instruction = (BASE_DIR / "context" / "2025-11-17-help_model.txt").read_text()

    retriever = await retriever_context(is_test=True) #
    parser = PydanticOutputParser(pydantic_object=ResponseFormatAi)
    format_instructions = parser.get_format_instructions()
    try:
        llm=ChatQwQ(
            api_key=config.QWEN_API_KEY,
            model=config.MODEL_DIALOG_AI,
            temperature=0.12,
            top_p=0.95,
            extra_body={
                "enable_thinking": True,
                "thinking_budget": 100,
            },
        )

        # Его задача: если вопрос зависит от истории, переформулировать его.
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{system_instruction}"),
                ("system", "Ты максимально кратко и по существу отвечаешь на вопросы пользователей, используя предоставленный контекст. Никогда не придумывай ответ самостоятельно."),
                # Сюда еще нужно передать совет по сну
                # Сюда сам сон
                ("system", "Используй следующие контекстные данные для ответа на вопрос: {context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        ).partial(format_instructions=format_instructions)


       
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Объединяем всё в RAG цепочку
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Получаем историю диалога пользователя из redis по ключу user_id
        final_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            RedisClient(
                session_id=f"{data.user_id}_{data.sleep_data}"
            ).get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        response = await final_rag_chain.ainvoke({"input": data.question})

        log.info(f"{response["context"]}")
        log.success(f"{data.user_id} QWEN response: {response['answer']}")
        return response['answer']
    except Exception as e:
        log.warning(f"QWEN generation failed: {e}")

        # await asyncio.sleep(1)


    # log.error(f"All attempts failed: {last_exception}")
    # return


if __name__ == "__main__":
    user_prompt_path = "/sleeptery/Sleeptery-AI/app/sleep_ai/resources/sleep.json"
    with open(user_prompt_path, "r") as f:
        sleep_data = f.read()

    asyncio.run(geration_pipe(
        data=UploadDialogAi(user_id=580, question="Я работаю до ночи, как мне победить бессоницу?", sleep_data="2025-12-16")
    ))
