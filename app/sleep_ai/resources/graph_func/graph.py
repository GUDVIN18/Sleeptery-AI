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
from .llm import main_llm, helper_llm, classifier_llm, SYSTEM_INSTRUCTION, HELP_MODEL_INSTRUCTION
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
        partial_variables={"system_instructions": HELP_MODEL_INSTRUCTION},
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
#     state.advice_type=AdviceType.ANALYSIS_ADVICE.value
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
    Определяет тип совета с помощью LLM
    """
    prompt = PromptTemplate(
        template="""
# Role

Ты классификатор советов Sleeptery.

Твоя задача — определить, нужно ли сегодня анализировать существующую привычку пользователя
или подобрать новую рекомендацию.

# Types

ANALYSIS_ADVICE

Выбирай, если:
- у пользователя есть конкретная привычка или ритуал;
- он действительно выполнял её;
- по данным последних дней можно оценить её связь со сном;
- анализ этой привычки актуален для сегодняшнего сна.

GENERATION_ADVICE

Выбирай, если:
- подходящей привычки нет;
- привычка есть, но недостаточно данных для её анализа;
- привычка не связана с текущей проблемой сна;
- существующая привычка не помогает и нужен другой подход.

# Main Rule

Ориентируйся прежде всего на сегодняшний сон относительно последних дней.

Наличие привычки само по себе НЕ является причиной выбирать ANALYSIS_ADVICE.

Сначала определи, что изменилось в сегодняшнем сне, затем проверь,
есть ли среди привычек пользователя релевантная этому изменению привычка.

Если такой связи нет — выбирай GENERATION_ADVICE.

# User Data

## Привычки

Стандартные:
{default_habits}

Пользовательские:
{custom_habits}

## Сон за последние дни

{sleep_weekly_stats}

## Сон сегодня

{sleep_daily_stats}

# Output

{format_instructions}

Верни только JSON.
""",
        input_variables=[
            "default_habits",
            "custom_habits",
            "sleep_daily_stats",
            "sleep_weekly_stats",
            "user_goal",
        ],
        partial_variables={
            "format_instructions": classifier_parser.get_format_instructions()
        }
    )

    chain = prompt | classifier_llm | classifier_parser

    habits = extract_habits(state.sleep_json["user_diary_records"])
    default_habits = habits["default_habits"]
    custom_habits = habits["custom_habits"]
    # user_goal = await SleepteryDairyAPI.get_user_goal(user_id=state.user_id, date=state.sleep_date)
    user_goal = None

    log.debug(f"{user_goal=}|{default_habits=}|{custom_habits=}")

    result = await chain.ainvoke({
        "default_habits": default_habits,
        "custom_habits": custom_habits,
        "sleep_daily_stats": state.sleep_daily_stats,
        "sleep_weekly_stats": state.sleep_weekly_stats,
        "user_goal": user_goal
    })

    state.advice_type = result["advice_type"]

    log.info(f"Определили тип будущего совета: {state.advice_type}")

    return state

def route_advice(state: SleepGraphAi) -> str:
    if state.advice_type == AdviceType.ANALYSIS_ADVICE.value:
        return AdviceType.ANALYSIS_ADVICE.value
    if state.advice_type == AdviceType.GENERATION_ADVICE.value:
        return AdviceType.GENERATION_ADVICE.value
    return AdviceType.GENERATION_ADVICE.value

@current_time
async def llm_analysis_response(state: SleepGraphAi) -> SleepGraphAi:
    """Совет на основе уже существующего ритуала"""

    prompt_template = PromptTemplate(
        template="""
{system_instructions}
\n\n\n
Важно!!!
Пользователь УЖЕ использует ритуал перед сном.

Дневник пользователя:
{user_diary_records}

Данные сна за сегодня:
{sleep_daily_stats}

{format_instructions}

Верни ТОЛЬКО JSON без комментариев.
Не используй английские слова.
""",
        input_variables=[
            "user_diary_records",
            "sleep_daily_stats",
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "system_instructions": SYSTEM_INSTRUCTION
        }
    )

    chain = prompt_template | main_llm | parser

    result = await chain.ainvoke({
        "user_diary_records": state.user_diary_records,
        "sleep_daily_stats": state.sleep_daily_stats
    })
    state.sleep_assessment = result["sleep_assessment"]
    state.response = result["response"]
    state.diary_recommendation = result.get("diary_recommendation")
    state.mission = result.get("mission")
    log.debug(f"Завершили создание совета на основе ритуала/миссии")
    return state

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