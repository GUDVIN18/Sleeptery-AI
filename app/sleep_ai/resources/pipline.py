from langgraph.graph import StateGraph, START, END
from pathlib import Path
from typing import Dict, Optional
import asyncio
import json
from langchain_qwq import ChatQwQ
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.globals import set_debug
from app.include.config import config
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_qwq import ChatQwQ
from langchain.agents import create_agent
from .schemas.sleepai import (
    ResponseFormat, 
    ResponseFormatAi, 
    UploadSleepAi, 
    SleepGraphAi,
    AdviceType,
    AdviceClassifier,
    AdviceLLMResponse
)
from .RAG.rag_langchain import retrieve_context
from .exceptions import (
    SleepAiErrorGeneration,
    SleepAiErrorFormat, 
    SleepAiErrorConnect
)
from app.include.logging_config import logger as log


BASE_DIR = Path(__file__).resolve().parent.parent
set_debug(False)
try:
    SYSTEM_INSTRUCTION = (BASE_DIR / "context" / "2025-11-12-instruction.txt").read_text(encoding="utf-8")
    HELP_MODEL_INSTRUCTION = (BASE_DIR / "context" / "2025-11-17-help_model.txt").read_text(encoding="utf-8")
except FileNotFoundError as e:
    log.error(f"Failed to load prompt templates: {e}")
    raise

# def extract_full_block(block: Dict[str, dict | None]) -> Dict[str, Dict[str, Optional[str]]]:
#     result = {}
#     for key, value in block.items():
#         if value is None:
#             result[key] = {
#                 "amount": None,
#                 "type": None,
#                 "description": None
#             }
#             continue

#         result[key] = {
#             "amount": value.get("amount"),
#             "type": value.get("type"),
#             "description": value.get("description")
#         }
#     return result

def extract_full_block(block: Dict[str, dict | None]) -> Dict[str, Dict[str, Optional[str]]]:
    result = {}

    for key, value in block.items():

        # если значение None
        if value is None:
            result[key] = {
                "amount": None,
                "type": None,
                "description": None
            }
            continue

        # если это корректный dict
        if isinstance(value, dict):
            result[key] = {
                "amount": value.get("amount"),
                "type": value.get("type"),
                "description": value.get("description")
            }
            continue

        # если это строка/число/boolean
        result[key] = {
            "amount": value,
            "type": type(value).__name__,
            "description": None
        }

    return result


async def geration_pipe(sleep_data: UploadSleepAi) -> AdviceLLMResponse:
    if not config.QWEN_API_KEY:
        raise SleepAiErrorConnect("API key is not set.")
    
    user_diary_records = extract_full_block(sleep_data["user diary records"])
    sleep_daily_stats = extract_full_block(sleep_data["sleep daily stats"])
    sleep_weekly_stats = {
        date: extract_full_block(day)
        for date, day in sleep_data["sleep daily stats"].items()
    }
    history_sleep_assessment = [
        extract_full_block(day)
        for day in sleep_data["history sleep assessment"]
    ]
    graph = StateGraph(SleepGraphAi)

    agent_helper=create_agent(
        model=ChatQwQ(
            api_key=config.QWEN_API_KEY,
            model=config.MODEL_SLEEP_AI,
            temperature=0.05,
            top_p=0.95,
            extra_body={
                "enable_thinking": True,
                "thinking_budget": 100,
            },
        ),
        system_prompt=HELP_MODEL_INSTRUCTION,
        response_format=ResponseFormat
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

    classifier_llm = ChatQwQ(
        api_key=config.QWEN_API_KEY,
        model=config.MODEL_SLEEP_AI,
        temperature=0.1,
        extra_body={
            "enable_thinking": True,
            "thinking_budget": 30,
        }
    )

    # parser = JsonOutputParser(pydantic_object=SleepGraphAi)
    parser = JsonOutputParser(pydantic_object=AdviceLLMResponse)
    classifier_parser = JsonOutputParser(
        pydantic_object=AdviceClassifier
    )
    
    async def init_models(state: SleepGraphAi) -> SleepGraphAi:
        """Узле для сборки модели SleepGraphAi"""
        log.info(f"Начали собирать данные")
        state.user_diary_records=user_diary_records
        state.sleep_daily_stats=sleep_daily_stats
        state.sleep_weekly_stats=sleep_weekly_stats
        state.history_sleep_assessment=history_sleep_assessment
        return state
    
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
        problems = helper_analytics['structured_response'].analysis
        log.info(f"Extracted problems: {problems} ")
        rag_answer = await retrieve_context(topics=problems, is_test=True) # is_test=config.TEST_MODE_DB
        state.context_vector_db=rag_answer
        log.info(f"Нашли context_vector_db и завершили llm_search")
        return state


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

    def build_button(mission: Optional[str]) -> Optional[str]:
        if not mission:
            return None
        return f"Добавить {mission} в дневник"
    
    def extract_habits(block: dict):
        default_habits = block.get("default_habits")
        custom_habits = block.get("custom_habits")

        return {
            "default_habits": default_habits if default_habits else None,
            "custom_habits": custom_habits if custom_habits else None,
        }
    
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

        habits = extract_habits(sleep_data["user diary records"])
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
        if state.advice_type == AdviceType.ANALYSIS_ADVICE:
            return "analysis_response"
        if state.advice_type == AdviceType.GENERATION_ADVICE:
            return "generation_response"
        return "generation_response"
    
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
        state.button = build_button(state.mission)
        log.debug(f"Завершили создание совета на основе ритуала/миссии")
        return state
    
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
        state.button = build_button(state.mission)
        log.debug(f"Завершили создание нового совета")
        return state



    # добавляем node (узлы = наши функции)
    graph.add_node("init_models", init_models)
    graph.add_node("llm_search", llm_search)
    # graph.add_node("llm_analysis", llm_analysis)
    graph.add_node("llm_advice_classifier", llm_advice_classifier)
    graph.add_node("analysis_response", llm_analysis_response)
    graph.add_node("generation_response", llm_generation_response)

    # Теперь выстраиваем ребра (последоваельность)
    graph.add_edge(START, "init_models")
    graph.add_edge("init_models", "llm_search")
    graph.add_edge("llm_search", "llm_advice_classifier")
    graph.add_conditional_edges(
        "llm_advice_classifier",
        route_advice,
        {
            "analysis_response": "analysis_response",
            "generation_response": "generation_response"
        }
    )
    graph.add_edge("analysis_response", END)
    graph.add_edge("generation_response", END)
    app = graph.compile()

    initial_state = SleepGraphAi()
    result = await app.ainvoke(initial_state)
    result_format_ai = AdviceLLMResponse(
        **result
    )
    log.success(f"{result_format_ai=}")
    return result_format_ai


if __name__ == "__main__":
    user_prompt_path = "/sleeptery/Sleeptery-AI/app/sleep_ai/resources/sleep.json"
    with open(user_prompt_path, "r") as f:
        sleep_data = f.read()

    asyncio.run(geration_pipe(sleep_data=json.loads(sleep_data)))
