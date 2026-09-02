from langgraph.graph import StateGraph, START, END
import asyncio
import json
import traceback
import time
from app.include.config import config
from app.include.logging_config import logger as log
from .schemas.sleepai import (
    UploadSleepAi, 
    SleepGraphAi,
    AdviceType,
    AppVersion
)
import datetime as dt
from .exceptions import (
    SleepAiErrorGeneration,
    SleepAiErrorFormat, 
    SleepAiErrorConnect
)
from .graph_func.graph import (
    init_models,
    llm_search,
    llm_advice_classifier,
    llm_analysis_response,
    llm_generation_response,
    route_advice
)
from st_bases.telegram import TgLog
from confluent_kafka import Producer

     
producer = Producer({
    'bootstrap.servers': config.KAFKA_BROKER_URL,
    'client.id': 'sleep_ai_ready_generation',
    'acks': 'all',
    'enable.idempotence': True, # гарантирует, что сообщения не будут потеряны и не будут продублированы в случае сбоев,
    'compression.type': 'zstd', # сжатие сообщений для оптимизации производительности
})

async def geration_pipe(data: UploadSleepAi) -> SleepGraphAi | None:
    if not config.QWEN_API_KEY:
        raise SleepAiErrorConnect("API key is not set.")

    # if data.sleep_date < dt.date.today():
    #     log.info("Совет не будет сгенерирован. Сон не за сегодня")
    #     return None

    start_time = time.time()
    graph = StateGraph(SleepGraphAi)
    # узлы
    graph.add_node("init_models", init_models)
    graph.add_node("llm_search", llm_search)
    # graph.add_node("llm_analysis", llm_analysis)
    graph.add_node("llm_advice_classifier", llm_advice_classifier)
    graph.add_node("analysis_response", llm_analysis_response)
    graph.add_node("generation_response", llm_generation_response)

    # ребра
    graph.add_edge(START, "init_models")
    graph.add_edge("init_models", "llm_search")
    graph.add_edge("llm_search", "llm_advice_classifier")
    graph.add_conditional_edges(
        "llm_advice_classifier",
        route_advice,
        {
            AdviceType.ANALYSIS_ADVICE.value: "analysis_response",
            AdviceType.GENERATION_ADVICE.value: "generation_response"
        }
    )
    graph.add_edge("analysis_response", END)
    graph.add_edge("generation_response", END)
    app = graph.compile()

    try:
        initial_state = SleepGraphAi(**data.model_dump())
        result = await app.ainvoke(initial_state)
        result = SleepGraphAi(**result)
        log.debug(f"{result=}")

        end_time = time.time()
        log.success(f"{data.user_id}: SLLEP_AI Pipeline execution time: {end_time - start_time:.2f} seconds")
        producer.produce(
            topic="sleep_ai_ready_generation",
            key=f"{result.user_id}", # для каждого пользователя совет будет в одном партиции
            value=result.model_dump_json().encode('utf-8')
        )
        producer.flush()
        return result
    except Exception:
        await TgLog.error(f"SLLEP_AI Pipeline error: {traceback.format_exc()}")
        raise


# if __name__ == "__main__":
#     user_prompt_path = "/sleeptery/Sleeptery-AI/app/sleep_ai/resources/sleep_debug.json"
#     with open(user_prompt_path, "r") as f:
#         sleep_json = f.read()
#         data_test = UploadSleepAi(
#             app_version="dev",
#             user_id=123,
#             sleep_date="2024-10-01",
#             sleep_json=json.loads(sleep_json)
#         )
#     asyncio.run(geration_pipe(data=data_test))
