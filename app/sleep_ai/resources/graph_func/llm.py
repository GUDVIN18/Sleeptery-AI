import os
from pathlib import Path
from app.include.config import config
from langchain_qwq import ChatQwQ
from langchain.agents import create_agent
from langchain_core.globals import set_debug
from app.include.logging_config import logger as log
from app.sleep_ai.resources.schemas.sleepai import (
    ResponseFormat,
)
from langfuse.langchain import CallbackHandler


BASE_DIR = Path(__file__).resolve().parent.parent.parent
set_debug(False)
try:
    SYSTEM_INSTRUCTION = (BASE_DIR / "context" / "2025-11-12-instruction.txt").read_text(encoding="utf-8")
    HELP_MODEL_INSTRUCTION = (BASE_DIR / "context" / "2025-11-17-help_model.txt").read_text(encoding="utf-8")
except FileNotFoundError as e:
    log.error(f"Failed to load prompt templates: {e}")
    raise

os.environ["LANGFUSE_PUBLIC_KEY"] = config.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = config.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = config.LANGFUSE_BASE_URL


langfuse_handler = CallbackHandler()

agent_helper=create_agent(
    model=ChatQwQ(
        api_key=config.QWEN_API_KEY,
        model=config.MODEL_SLEEP_AI,
        temperature=0.1,
        top_p=0.9001,
        extra_body={
            "enable_thinking": False,
        },
        callbacks=[langfuse_handler],
    ),
    system_prompt=HELP_MODEL_INSTRUCTION,
    response_format=ResponseFormat
)


main_llm=ChatQwQ(
    api_key=config.QWEN_API_KEY,
    model=config.MODEL_DIALOG_AI,
    temperature=0.35,
    top_p=0.9001,
    extra_body={
        "enable_thinking": True,
        "thinking_budget": 250,
    },
    callbacks=[langfuse_handler],
)

classifier_llm = ChatQwQ(
    api_key=config.QWEN_API_KEY,
    model=config.MODEL_SLEEP_AI,
    temperature=0.1,
    extra_body={
        "enable_thinking": True,
        "thinking_budget": 30,
    },
    callbacks=[langfuse_handler],
)