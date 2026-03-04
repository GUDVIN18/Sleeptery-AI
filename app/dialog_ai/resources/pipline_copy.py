from pathlib import Path
import asyncio
import json
import os
import traceback
from typing import List
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
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .redis_client import RedisClient
from .RAG.rag_langchain import retriever_context
from .schemas.dialog import (
    ResponseFormatAi, 
    UploadDialogAi, 
    ResponseDialogAi
)
from .exceptions import (
    DialogAiErrorConnect,
    DialogAiErrorGeneration,
    DialogAiErrorFormat,
    DialogAiContentBlocked
)
from langchain_qdrant import QdrantVectorStore
from app.include.logging_config import logger as log
from app.include.config import config

    
BASE_DIR = Path(__file__).resolve().parent.parent
try:
    SYSTEM_INSTRUCTION = (BASE_DIR / "context" / "2025-12-12-instruction.txt").read_text(encoding="utf-8")
    CONTEXT_PROMPT = (BASE_DIR / "context" / "2026-02-20-contextualize_prompt.txt").read_text(encoding="utf-8")
except Exception as e:
    log.error(f"Failed to load prompts: {e}")

async def geration_pipe(
        data: UploadDialogAi,
        is_test: bool = config.TEST_MODE_DB
) -> ResponseDialogAi:
    if not config.QWEN_API_KEY:
        raise DialogAiErrorConnect("API key is not set.")
   
    # Получаем историю
    chat_history_obj = RedisClient(
        session_id=f"{data.user_id}_{data.sleep_date}"
    ).get_session_history_v2(
        user_id=data.user_id,
        sleep_date=data.sleep_date
    )
    current_history = chat_history_obj
    log.info(f"[{data.user_id}] History loaded: {len(current_history)} messages.")

    llm_helper = ChatQwQ(
        api_key=config.QWEN_API_KEY,
        model=config.MODEL_DIALOG_AI,
        temperature=0.1,
        top_p=0.95,
        extra_body={
            "enable_thinking": True,
            "thinking_budget": 120,
        },
    )
    main_llm=ChatQwQ(
        api_key=config.QWEN_API_KEY,
        model=config.MODEL_DIALOG_AI,
        temperature=0.4,
        top_p=0.95,
        extra_body={
            "enable_thinking": True,
            "thinking_budget": 350,
        },
    )
    main_llm.with_structured_output(ResponseFormatAi)

    try:
        context_prompt_helper = ChatPromptTemplate.from_messages([
            ("system", CONTEXT_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
    
        chain_context = context_prompt_helper | llm_helper
        result = await chain_context.ainvoke({
            "chat_history": current_history,
            "input": data.message
        })
        search_query = result.content
        log.info(f"[{data.user_id}] Rewritten query: {search_query}")

        # Поиск в векторной базе (Ретривер)
        log.info(f"[{data.user_id}] Searching vector DB...")
        retriever = await retriever_context(is_test=is_test)
        docs = await retriever.ainvoke(search_query)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        log.info(f"[{data.user_id}] Found {len(docs)} relevant documents.\n\n{context_text=}")


        if data.sleep_json:
            sleep_data_str = json.dumps(data.sleep_json, ensure_ascii=False).replace("{", "").replace("}", "")
        else:
            sleep_data_str="Не предоставлены"


        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", f"Данные сна (ПЕРЕВЕДИ В ЧАСЫ): {sleep_data_str}"),
            ("system", f"Рекомендация по улучшению сна: {data.sleep_assessment or 'Не предоставлен'}"),
            ("system", SYSTEM_INSTRUCTION),
            ("system", "Контекст из базы знаний для ответа на вопрос: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        chain_context = qa_prompt | main_llm 
        response = await chain_context.ainvoke({
            "context": context_text,
            "chat_history": current_history,
            "input": data.message
        })
    except Exception as e:
        if "data_inspection_failed" in str(e):
            log.error(f"Content blocked")
            raise DialogAiContentBlocked
        else:
            log.error(f"Error during generation: \n{traceback.format_exc()}")
            raise DialogAiErrorGeneration

    log.info(f"Final answer: {response}")

    if response is None:
        log.error(f"Error response is None: \n{traceback.format_exc()}")
        raise DialogAiErrorGeneration

    RedisClient(
        session_id=f"{data.user_id}_{data.sleep_date}"
    ).add_message(
        role="user",
        message=data.message
    )
    
    RedisClient(
        session_id=f"{data.user_id}_{data.sleep_date}"
    ).add_message(
        role="ai",
        message=response.content
    )

    return ResponseDialogAi(
        answer=response.content
    )

if __name__ == "__main__":
    user_prompt_path = "app/dialog_ai/resources/dialog.json"
    with open(user_prompt_path, "r") as f:
        sleep_json = json.load(f)

    asyncio.run(geration_pipe(
        data=UploadDialogAi(
            user_id=5801, 
            message="Привет, как улучшить сон?", 
            sleep_json=None,
            sleep_assessment=None,
            sleep_date="2026-02-27"),
        is_test=True
        )
    )

