from app.include.logging_config import logger as log
import asyncio
from app.sleep_ai.resources.schemas.sleepai import (
    SleepGraphAi,
    AdviceType,
)
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .parcers import extract_full_block, parser, classifier_parser
from .llm import (
    main_llm, 
    helper_llm, 
    classifier_llm, 
    SYSTEM_INSTRUCTION, 
    SEARCH_MODEL_INSTRUCTION,
    GOAL_INSTRUCTION,
    ADVICE_CLASSIFIER_INSTRUCTION,
    RITUAL_INSTRUCTION
)
from ..RAG.rag_langchain import retrieve_context
from app.include.decorator import current_time
from app.sleep_ai.resources.sleeptery_api import SleepteryDairyAPI


@current_time
async def init_models(state: SleepGraphAi) -> SleepGraphAi:
    """Узле для сборки модели SleepGraphAi"""
    log.info(f"Начали собирать данные")
    user_diary_records = extract_full_block(state.sleep_json["user_diary_records"])
    sleep_daily_stats = extract_full_block(state.sleep_json["sleep_daily_stats"])
    sleep_weekly_stats = {
        date: extract_full_block(day)
        for date, day in state.sleep_json["sleep_daily_stats"].items()
    }
    history_sleep_assessment = [
        extract_full_block(day)
        for day in state.sleep_json["history_sleep_assessment"]
    ]
    rituals = (
        (user_diary_records.get("default_habits") or [])
        + (user_diary_records.get("custom_habits") or [])
    )

    state.user_rituals=rituals
    state.user_diary_records=user_diary_records
    state.sleep_daily_stats=sleep_daily_stats
    state.sleep_weekly_stats=sleep_weekly_stats
    state.history_sleep_assessment=None # Не будем пока передавать историю советов в LLM, чтобы не было повторов.
    print("Завершили сбор данных и перешли к llm_search")
    return state

@current_time
async def llm_search(state: SleepGraphAi) -> SleepGraphAi:
    """Узел для анализа проблемы и поика в векторной БД"""
    prompt = PromptTemplate(
        template="""
{system_instructions}

Дневник пользователя на сегодняшний день:
{user_diary_records}

Сформированный сон за сегодня:
{sleep_daily_stats}

Сформированный сон за последние 3 дня:
{sleep_weekly_stats}

История советов по улучшению сна:
{history_sleep_assessment}

Верни только JSON-массив строк. Без markdown, комментариев и дополнительных полей.
""",
        input_variables=[
            "user_diary_records",
            "sleep_daily_stats",
            "sleep_weekly_stats",
            "history_sleep_assessment",
        ],
        partial_variables={"system_instructions": SEARCH_MODEL_INSTRUCTION},
    )
    chain = prompt | helper_llm | JsonOutputParser()

    try:
        problems = await asyncio.wait_for(
            chain.ainvoke({
                "user_diary_records": state.user_diary_records,
                "sleep_daily_stats": state.sleep_daily_stats,
                "sleep_weekly_stats": state.sleep_weekly_stats,
                "history_sleep_assessment": state.history_sleep_assessment,
            }),
            timeout=90,
        )
    except Exception:
        log.exception("Ошибка helper LLM в llm_search. Продолжаем без тем для RAG.")
        problems = []

    if isinstance(problems, dict):
        problems = problems.get("recommendations", [])
    if not isinstance(problems, list):
        log.warning(f"Некорректный формат helper LLM: {problems!r}")
        problems = []
    problems = [problem for problem in problems if isinstance(problem, str)]

    log.info(f"Extracted problems: {problems} ")
    try:
        rag_answer = await asyncio.wait_for(
            retrieve_context(topics=problems, is_test=True), # is_test=config.TEST_MODE_DB
            timeout=90,
        )
    except Exception:
        log.exception("Ошибка RAG в llm_search. Продолжаем без контекста из векторной БД.")
        rag_answer = []
    state.context_vector_db=rag_answer
    return state

# @current_time
# async def llm_analysis(state: SleepGraphAi) -> SleepGraphAi:
#     """Узел для анализа дневника: 
#     Если в дневнике есть привычка или ритуал, то: 
#         1) анализируем, как эта привычка соблюдается (Какой эффект оказывает этот ритуал при его соблюдении - учитывай фазы сна и оценку сна.)
#         2) добавь по ритуалу оценку - ПРИМЕР! молодец сегодня 'соблюдение ритуала пользователя' перед сном и видишь опять REM (например) выше среднего. 
#     Если в дневнике нет привычки или ритуала, то советуем ему новый, основываясь на базу знаний.
#     """
#     # тут по сути должна быть логика или ИИ которая формирует тип совета (генерация нового или оценка старого)
#     state.advice_type=AdviceType.RITUAL_ADVICE.value
#     return state


def extract_habits(block: dict):
    default_habits = block.get("default_habits")
    custom_habits = block.get("custom_habits")

    return {
        "default_habits": default_habits if default_habits else None,
        "custom_habits": custom_habits if custom_habits else None,
    }

@current_time
async def llm_advice_classifier(state: SleepGraphAi) -> SleepGraphAi:
    """
    Определяет тип совета с помощью LLM.
    """
    prompt = PromptTemplate(
        template="""
{system_instructions}

# User Data

## Цель пользователя
{user_goal}

## Привычки
{user_rituals}

## Сон за последние дни
{sleep_weekly_stats}

## Сон сегодня
{sleep_daily_stats}


# Output
{format_instructions}

Верни только JSON.
""",
        input_variables=[
            "sleep_daily_stats",
            "sleep_weekly_stats",
            "user_goal",
            "user_rituals"
        ],
        partial_variables={
            "format_instructions": classifier_parser.get_format_instructions(),
            "system_instructions": ADVICE_CLASSIFIER_INSTRUCTION
        }
    )

    chain = prompt | classifier_llm | classifier_parser

    user_goal = await SleepteryDairyAPI.get_user_goal(
        user_id=state.user_id,
        date=state.sleep_date,
    )

    log.debug(
        f"{user_goal=}"
    )

    result = await asyncio.wait_for(
        chain.ainvoke({
            "sleep_daily_stats": state.sleep_daily_stats,
            "sleep_weekly_stats": state.sleep_weekly_stats,
            "user_goal": user_goal,
            "user_rituals": state.user_rituals
        }),
        timeout=90,
    )
    state.user_goal = user_goal
    state.advice_type = result["advice_type"]

    log.info(f"Определили тип будущего совета: {state.advice_type}")

    return state

def route_advice(state: SleepGraphAi) -> str:
    if state.advice_type == AdviceType.RITUAL_ADVICE.value:
        return AdviceType.RITUAL_ADVICE.value
    if state.advice_type == AdviceType.GENERATION_ADVICE.value:
        return AdviceType.GENERATION_ADVICE.value
    if state.advice_type == AdviceType.GOAL_ADVICE.value:
        return AdviceType.GOAL_ADVICE.value
    return AdviceType.GENERATION_ADVICE.value

@current_time
async def llm_ritual_response(state: SleepGraphAi) -> SleepGraphAi:
    """Совет на основе уже существующего ритуала"""

    prompt_template = PromptTemplate(
        template="""
{system_instructions}
\n\n\n
{ritual_system}

# Ритуалы пользователя
{user_rituals}

# Дневник пользователя
{user_diary_records}

# Сон за сегодня:
{sleep_daily_stats}

# История советов (новый не повторять со старыми):
{history_sleep_assessment}

# Формат ответа
{format_instructions}

Верни ТОЛЬКО JSON без комментариев.
Не используй английские слова.
""",
        input_variables=[
            "user_diary_records",
            "sleep_daily_stats",
            "context",
            "user_rituals",
            "history_sleep_assessment"
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "system_instructions": SYSTEM_INSTRUCTION,
            "ritual_system": RITUAL_INSTRUCTION,
        }
    )

    chain = prompt_template | main_llm | parser

    try:
        result = await asyncio.wait_for(
            chain.ainvoke({
                "user_diary_records": state.user_diary_records,
                "sleep_daily_stats": state.sleep_daily_stats,
                "user_rituals": state.user_rituals,
                "context": state.context_vector_db,
                "history_sleep_assessment": state.history_sleep_assessment,
            }),
            timeout=90,
        )
        state.sleep_assessment = result["sleep_assessment"]
        state.response = result["response"]
        state.diary_recommendation = result.get("diary_recommendation")
        state.mission = result.get("mission")
        log.debug(f"Завершили создание нового совета")

        return state
    except Exception as e:
        log.exception(f"Ошибка main LLM в llm_generation_response: {e}")
        raise

@current_time
async def llm_generation_response(state: SleepGraphAi) -> SleepGraphAi:
    """Генерация нового совета"""

    prompt_template = PromptTemplate(
        template="""
{system_instructions}

Данные сна за сегодня:
{sleep_daily_stats}

История советов (новый не повторять со старыми):
{history_sleep_assessment}

Контекст базы знаний:
{context}

Сон за последние дни:
{sleep_weekly_stats}

{format_instructions}

Верни ТОЛЬКО JSON без комментариев.
Не используй английские слова.
""",
        input_variables=[
            "sleep_daily_stats",
            "history_sleep_assessment",
            "context",
            "sleep_weekly_stats"
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "system_instructions": SYSTEM_INSTRUCTION
        }
    )

    chain = prompt_template | main_llm | parser
    try:
        result = await asyncio.wait_for(
            chain.ainvoke({
                "sleep_daily_stats": state.sleep_daily_stats,
                "history_sleep_assessment": state.history_sleep_assessment,
                "context": state.context_vector_db,
                "sleep_weekly_stats": state.sleep_weekly_stats
            }),
            timeout=90,
        )

        state.sleep_assessment = result["sleep_assessment"]
        state.response = result["response"]
        state.diary_recommendation = result.get("diary_recommendation")
        state.mission = result.get("mission")
        log.debug(f"Завершили создание нового совета")

        return state
    except Exception as e:
        log.exception(f"Ошибка main LLM в llm_generation_response: {e}")
        raise


@current_time
async def llm_goal_response(state: SleepGraphAi) -> SleepGraphAi:
    """Генерация нового совета"""

    prompt_template = PromptTemplate(
        template="""
{system_instructions}\n\n\n
{goal_instructions}

Данные сна за сегодня:
{sleep_daily_stats}

История советов (новый не повторять со старыми):
{history_sleep_assessment}

Контекст базы знаний:
{context}

Пользовательская цель:
{user_goal}

{format_instructions}

Верни ТОЛЬКО JSON без комментариев.
Не используй английские слова.
""",
        input_variables=[
            "sleep_daily_stats",
            "history_sleep_assessment",
            "context",
            "user_goal"
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "system_instructions": SYSTEM_INSTRUCTION,
            "goal_instructions": GOAL_INSTRUCTION
        }
    )

    chain = prompt_template | main_llm | parser
    try:
        result = await asyncio.wait_for(
            chain.ainvoke({
                "sleep_daily_stats": state.sleep_daily_stats,
                "history_sleep_assessment": state.history_sleep_assessment,
                "context": state.context_vector_db,
                "user_goal": state.user_goal
            }),
            timeout=90,
        )

        state.sleep_assessment = result["sleep_assessment"]
        state.response = result["response"]
        state.diary_recommendation = result.get("diary_recommendation")
        state.mission = result.get("mission")
        log.debug(f"Завершили создание нового совета")

        return state
    except Exception as e:
        log.exception(f"Ошибка main LLM в llm_generation_response: {e}")
        raise