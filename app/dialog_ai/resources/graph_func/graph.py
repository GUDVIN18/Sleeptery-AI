from langchain.agents import create_agent
import json
import time
from app.dialog_ai.resources.redis_async_client import AsyncRedisClient
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.include.logging_config import logger as log
from .llm import main_llm, llm_analytics, SYSTEM_INSTRUCTION, CONTEXT_PROMPT
from .parcers import (
    parser_main_llm,
    parser_helper_llm,
)
from ..RAG.rag_langchain import retriever_context
from ..schemas.dialog import (
    DialogAi
)


def current_time(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        log.info(f"Время выполнения функции {func.__name__}: {end_time - start_time:.2f} секунд")
        return result
    return wrapper


@current_time
async def _current_history(state: DialogAi) -> DialogAi:
    """Узел для получения истории сообщений"""
    # Соединение само закроется, когда мы выйдем из блока async with
    async with AsyncRedisClient(session_id=f"{state.user_id}_{state.sleep_date}") as client:
        current_history = await client.get_session_history_v2(
            user_id=state.user_id,
            sleep_date=state.sleep_date
        )
    state.history_messages = current_history[-20:]
    log.debug(f"{state.user_id}: История подгружена. Всего {len(current_history)} сообщений.")
    return state

@current_time
async def llm_helper(state: DialogAi) -> DialogAi:
    """Узел для анализа проблемы пользователя (для RAG)"""
    prompt_template_helper = PromptTemplate(
        template="""
    {system_instructions}

    
    История диалога:
    {chat_history}

    Вопрос пользователя:
    {input}

    {format_instructions}

    Верни ТОЛЬКО JSON без дополнительных комментариев! 
    не допускай использование английскийх слов в ответе
    """,
        input_variables=[
            "chat_history",
            "input",
        ],
        partial_variables={
            "system_instructions": CONTEXT_PROMPT,
            "format_instructions": parser_helper_llm.get_format_instructions(),
        }
    )

    chain = prompt_template_helper | llm_analytics | parser_helper_llm
    result = await chain.ainvoke({
        "chat_history": state.history_messages,
        "input": state.message
    })
    state.context_rag_search = result['answer']
    log.info(f"{state.user_id}: Проблема для поиска в бд: {result['answer']}")
    return state

@current_time
async def search_vector_db(state: DialogAi) -> DialogAi:
    """Узел для поиска докуметов в векторной БД"""
    retriever = await retriever_context(is_test=state.test_mode)
    docs = await retriever.ainvoke(state.context_rag_search)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    state.context_vector_db = context_text
    log.debug(f"{state.user_id}: Найдено {len(docs)} документов")
    return state

@current_time
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
            "format_instructions": parser_main_llm.get_format_instructions(),
            "system_instructions": SYSTEM_INSTRUCTION
        }
    )

    chain = prompt_template | main_llm | parser_main_llm
    try:
        response = await chain.ainvoke({
            "sleep_json": sleep_data_str,
            "sleep_assessment": state.sleep_assessment or "Не предоставлен",
            "context": state.context_vector_db,
            "history": state.history_messages,
            "question": state.message
        })
        log.debug(f"{state.user_id}: {response=}")
        state.answer = response['answer']
        log.debug(f"{state.user_id}: Полученные кнопки из ответа: {response['buttons']=}")
        state.buttons = response['buttons']
    except Exception as e:
        log.error(f"{state.user_id}: Ошибка в llm_response: {e}")
    return state

