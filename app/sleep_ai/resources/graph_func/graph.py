from app.include.logging_config import logger as log
from app.sleep_ai.resources.schemas.sleepai import (
    SleepGraphAi,
    AdviceType,
)
from typing import Optional
from langchain_core.prompts import PromptTemplate
from .parcers import extract_full_block, parser, classifier_parser
from .llm import main_llm, agent_helper, classifier_llm, SYSTEM_INSTRUCTION
from ..RAG.rag_langchain import retrieve_context
from app.include.decorator import current_time


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
    state.history_sleep_assessment=history_sleep_assessment
    return state

@current_time
async def llm_search(state: SleepGraphAi) -> SleepGraphAi:
    """Узел для анализа проблемы и поика в векторной БД"""
    helper_analytics = await agent_helper.ainvoke(
        {"messages":
            [
                {"role": "user", "content": f"Дневник пользователя на сегодняшний день: {state.user_diary_records}"},
                {"role": "user", "content": f"Сформированный сон за сегодня: {state.sleep_daily_stats}"},
                {"role": "user", "content": f"Сформированный сон за последние 3 дня: {state.sleep_weekly_stats}"},
                {"role": "user", "content": f"История советов по улучшению сна: {state.history_sleep_assessment}"},
            ]
        }
    )
    problems = helper_analytics['structured_response'].recommendations
    log.info(f"Extracted problems: {problems} ")
    rag_answer = await retrieve_context(topics=problems, is_test=True) # is_test=config.TEST_MODE_DB
    state.context_vector_db=rag_answer
    log.info(f"Нашли context_vector_db и завершили llm_search")
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
Ты классификатор типов советов для улучшения сна.

Определи тип совета:

ANALYSIS_ADVICE
если пользователь уже выполняет ритуал/привычку перед сном и его нужно проанализировать.

GENERATION_ADVICE
если ритуала нет и нужно предложить новый или ритуал есть, но по анализу последних дней он не помогает.

Привычка или ритуал пользователя:
{default_habits} и {custom_habits}

Сон за последние дни:
{sleep_weekly_stats}

Cон за сегодня:
{sleep_daily_stats}

{format_instructions}

Верни только JSON.
""",
        input_variables=[
            # "user_diary_records",
            "default_habits",
            "custom_habits",
            "sleep_daily_stats",
            "sleep_weekly_stats"
        ],
        partial_variables={
            "format_instructions": classifier_parser.get_format_instructions()
        }
    )

    chain = prompt | classifier_llm | classifier_parser

    habits = extract_habits(state.sleep_json["user_diary_records"])
    default_habits = habits["default_habits"]
    custom_habits = habits["custom_habits"]
    log.debug(f"{default_habits=}|{custom_habits=}")
    result = await chain.ainvoke({
        # "user_diary_records": state.user_diary_records,
        "default_habits": default_habits,
        "custom_habits": custom_habits,
        "sleep_daily_stats": state.sleep_daily_stats,
        "sleep_weekly_stats": state.sleep_weekly_stats,
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

Пользователь уже использует ритуал перед сном.

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
    result = await chain.ainvoke({
        "sleep_daily_stats": state.sleep_daily_stats,
        "history_sleep_assessment": state.history_sleep_assessment,
        "context": state.context_vector_db,
        "sleep_weekly_stats": state.sleep_weekly_stats
    })

    state.sleep_assessment = result["sleep_assessment"]
    state.response = result["response"]
    state.diary_recommendation = result.get("diary_recommendation")
    state.mission = result.get("mission")
    log.debug(f"Завершили создание нового совета")
    return state