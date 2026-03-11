from langgraph.graph import StateGraph, START, END
import asyncio
import json
from app.include.config import config
from app.include.logging_config import logger as log
from .schemas.sleepai import (
    UploadSleepAi, 
    SleepGraphAi,
    AdviceType,
    AdviceLLMResponse
)
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


async def geration_pipe(sleep_data: UploadSleepAi) -> AdviceLLMResponse:
    if not config.QWEN_API_KEY:
        raise SleepAiErrorConnect("API key is not set.")
    
    graph = StateGraph(SleepGraphAi)
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
            AdviceType.ANALYSIS_ADVICE.value: "analysis_response",
            AdviceType.GENERATION_ADVICE.value: "generation_response"
        }
    )
    graph.add_edge("analysis_response", END)
    graph.add_edge("generation_response", END)
    app = graph.compile()

    initial_state = SleepGraphAi(sleep_data=sleep_data)
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
