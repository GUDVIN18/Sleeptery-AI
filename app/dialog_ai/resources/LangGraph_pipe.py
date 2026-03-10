from langgraph.graph import StateGraph, START, END
from pathlib import Path
import asyncio
import json
import os
import traceback
from typing import List
from langchain.agents import create_agent
from langchain_qwq import ChatQwQ
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.globals import set_debug
from .redis_client import RedisClient
from .RAG.rag_langchain import retriever_context
from .schemas.dialog import (
    ResponseFormatAi, 
    UploadDialogAi, 
    ResponseDialogAi,
    DialogAi
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
from .tool_calling import *
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate



BASE_DIR = Path(__file__).resolve().parent.parent
set_debug(False)
try:
    SYSTEM_INSTRUCTION = (BASE_DIR / "context" / "2025-12-12-instruction.txt").read_text(encoding="utf-8")
    CONTEXT_PROMPT = (BASE_DIR / "context" / "2026-02-20-contextualize_prompt.txt").read_text(encoding="utf-8")
except Exception as e:
    log.error(f"Failed to load prompts: {e}")

async def geration_pipe(
        data: UploadDialogAi,
        is_test: bool = config.TEST_MODE_DB
) -> DialogAi:
    if not config.QWEN_API_KEY:
        raise DialogAiErrorConnect("API key is not set.")
    graph = StateGraph(DialogAi)

    llm_analytics = ChatQwQ(
        api_key=config.QWEN_API_KEY,
        model=config.MODEL_DIALOG_AI,
        temperature=0.1,
        top_p=0.95,
        extra_body={
            "enable_thinking": True,
            "thinking_budget": 70,
        },
    )
    main_llm=ChatQwQ(
        api_key=config.QWEN_API_KEY,
        model=config.MODEL_DIALOG_AI,
        temperature=0.3,
        top_p=0.95,
        extra_body={
            "enable_thinking": True,
            "thinking_budget": 350,
        },
    )

    retriever = await retriever_context(is_test=is_test)
    parser = JsonOutputParser(pydantic_object=DialogAi)

    def _current_history(state: DialogAi) -> DialogAi:
        """Узел для получения истории сообщений"""
        current_history = RedisClient(
            session_id=f"{data.user_id}_{data.sleep_date}"
        ).get_session_history_v2(
            user_id=data.user_id,
            sleep_date=data.sleep_date
        )
        state.history_messages = current_history
        log.debug(f"{state.user_id}: История подгружена. Всего {len(current_history)} сообщений.")
        return state
    
    async def llm_helper(state: DialogAi) -> DialogAi:
        """Узел для анализа проблемы пользователя (для RAG)"""
        context_prompt_analytics = ChatPromptTemplate.from_messages([
            ("system", CONTEXT_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        chain_context = context_prompt_analytics | llm_analytics
        result = await chain_context.ainvoke({
            "chat_history": state.history_messages,
            "input": state.message
        })
        state.context_rag_search = result.content
        log.info(f"{data.user_id}: Проблема для поиска в бд: {result.content}")
        return state

    async def search_vector_db(state: DialogAi) -> DialogAi:
        """Узел для поиска докуметов в векторной БД"""
        docs = await retriever.ainvoke(state.context_rag_search)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        state.context_vector_db = context_text
        log.debug(f"{state.user_id}: Найдено {len(docs)} документов")
        return state
    
    async def llm_response(state: DialogAi) -> DialogAi:
        """Узел для ответа пользователю на вопрос по контексту из базы знаний"""
        if state.sleep_json:
            sleep_data_str = json.dumps(state.sleep_json, ensure_ascii=False).replace("{", "").replace("}", "")
        else:
            sleep_data_str="Не предоставлены"

        prompt_template = PromptTemplate(
            template="""
        {system_instructions}

        Данные сна пользователя:
        {sleep_json}

        Рекомендация по улучшению сна:
        {sleep_assessment}

        Контекст из базы знаний:
        {context}

        История диалога:
        {history}

        Вопрос пользователя:
        {question}

        {format_instructions}

        Верни ТОЛЬКО JSON без дополнительных комментариев! 
        не допускай использование английскийх слов в ответе
        """,
            input_variables=[
                "sleep_json",
                "sleep_assessment",
                "context",
                "history",
                "question",
            ],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "system_instructions": SYSTEM_INSTRUCTION
            }
        )

        chain = prompt_template | main_llm | parser

        response = await chain.ainvoke({
            "sleep_json": sleep_data_str,
            "sleep_assessment": state.sleep_assessment or "Не предоставлен",
            "context": state.context_vector_db,
            "history": state.history_messages,
            "question": state.message
        })
        log.debug(f"{state.user_id}: {response=}")
        state.answer = response['answer']
        state.button = response['button']
        return state



    # добавляем node (узлы = наши функции)
    graph.add_node("_current_history", _current_history)
    graph.add_node("llm_helper", llm_helper)
    graph.add_node("search_vector_db", search_vector_db)
    graph.add_node("llm_response", llm_response)


    # Теперь выстраиваем ребра (последоваельность)
    graph.add_edge(START, "_current_history")
    graph.add_edge("_current_history", "llm_helper")
    graph.add_edge("llm_helper", "search_vector_db")
    graph.add_edge("search_vector_db", "llm_response")
    graph.add_edge("llm_response", END)
    app = graph.compile()

    initial_state = DialogAi(**data.model_dump())

    result = await app.ainvoke(initial_state)

    RedisClient(
        session_id=f"{data.user_id}_{data.sleep_date}"
    ).add_message(
        role="user",
        message=result['message']
    )
    RedisClient(
        session_id=f"{data.user_id}_{data.sleep_date}"
    ).add_message(
        role="ai",
        message=result['answer']
    )
    return DialogAi(**result)




if __name__ == "__main__":
    user_prompt_path = "app/dialog_ai/resources/dialog.json"
    with open(user_prompt_path, "r") as f:
        sleep_json = json.load(f)
    sleep_assessment="Восстановительные циклы показывают прогресс — REM-фаза крепнет третью ночь подряд, хотя график сна всё ещё ищет точку опоры.",
    response_ass="RAG-наука подтверждает: утренний свет — самый мощный якорь для внутренних часов. Яркий свет в первые 30 минут после пробуждения запускает кортизол и помогает телу понять — день начался. Попробуй миссию «Световой якорь»: сразу после подъёма открой шторы или включи яркий свет на 5–10 минут. Это сигнал для мозга: пора просыпаться, и вечером мелатонин придёт вовремя. Внеси утренний световой ритуал в дневник — так Sleeptery сможет отслеживать, как яркий свет после пробуждения влияет на твоё вечернее засыпание и стабильность графика.",
    asyncio.run(geration_pipe(
        data=UploadDialogAi(
            user_id=58055, 
            message="Хочу новый совет", 
            sleep_json=sleep_json,
            sleep_assessment=f"{sleep_assessment}\n{response_ass}",
            sleep_date="2026-02-27"),
        is_test=True
        )
    )
