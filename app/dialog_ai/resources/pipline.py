import json
import time
import asyncio
from langgraph.graph import StateGraph, START, END
from .redis_async_client import AsyncRedisClient
from app.include.logging_config import logger as log
from app.include.config import config
from .schemas.dialog import (
    UploadDialogAi, 
    DialogAi
)
from .exceptions import (
    DialogAiErrorConnect,
)
from .graph_func.graph import (
    _current_history,
    llm_helper,
    search_vector_db,
    llm_response,
)


async def geration_pipe(
        data: UploadDialogAi,
        is_test: bool = config.TEST_MODE_DB
) -> DialogAi:
    if not config.QWEN_API_KEY:
        raise DialogAiErrorConnect("API key is not set.")
    start_time = time.time()
    graph = StateGraph(DialogAi)

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

    initial_state = DialogAi(**data.model_dump(), test_mode=is_test)

    result = await app.ainvoke(initial_state)
    if result:
        try:
            async with AsyncRedisClient(session_id=f"{data.user_id}_{data.sleep_date}") as client:
                await client.add_message(
                    role="user",
                    message=result['message']
                )
                await client.add_message(
                    role="ai",
                    message=result['answer']
                )
        except Exception as e:
            log.error(f"Ошибка в DialogAI при добавлении истории: {e}")
        end_time = time.time()
        log.success(f"{data.user_id}: Pipeline execution time: {end_time - start_time:.2f} seconds")
        return DialogAi(**result)




if __name__ == "__main__":
    user_prompt_path = "app/dialog_ai/resources/dialog.json"
    with open(user_prompt_path, "r") as f:
        sleep_json = json.load(f)
    sleep_assessment="Восстановительные циклы показывают прогресс — REM-фаза крепнет третью ночь подряд, хотя график сна всё ещё ищет точку опоры.",
    response_ass="RAG-наука подтверждает: утренний свет — самый мощный якорь для внутренних часов. Яркий свет в первые 30 минут после пробуждения запускает кортизол и помогает телу понять — день начался. Попробуй миссию «Световой якорь»: сразу после подъёма открой шторы или включи яркий свет на 5–10 минут. Это сигнал для мозга: пора просыпаться, и вечером мелатонин придёт вовремя. Внеси утренний световой ритуал в дневник — так Sleeptery сможет отслеживать, как яркий свет после пробуждения влияет на твоё вечернее засыпание и стабильность графика.",
    asyncio.run(geration_pipe(
        data=UploadDialogAi(
            user_id=580566, 
            message="В какой позе спать?", 
            sleep_json=sleep_json,
            sleep_assessment=f"{sleep_assessment}\n{response_ass}",
            sleep_date="2026-02-27"),
        is_test=True
        )
    )
