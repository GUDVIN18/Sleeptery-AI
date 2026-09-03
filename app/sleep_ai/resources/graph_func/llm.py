import os
from pathlib import Path
from app.include.config import config
from langchain_qwq import ChatQwQ
from langchain_core.globals import set_debug
from app.include.logging_config import logger as log
from langfuse.langchain import CallbackHandler


try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    HELP_MODEL_INSTRUCTION = (
        BASE_DIR / "context" / "help" / "system.md"
    ).read_text(encoding="utf-8")

    MAIN_SYSTEM_INSTRUCTION = (
        BASE_DIR / "context" / "main" / "system.md"
    ).read_text(encoding="utf-8")

    MAIN_STYLE_INSTRUCTION = (
        BASE_DIR / "context" / "main" / "style.md"
    ).read_text(encoding="utf-8")

    SYSTEM_INSTRUCTION = f"{MAIN_SYSTEM_INSTRUCTION}\n\n\n{MAIN_STYLE_INSTRUCTION}"
except Exception as e: 
    log.error(f"Error prompt {e}")
    raise
os.environ["LANGFUSE_PUBLIC_KEY"] = config.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = config.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = config.LANGFUSE_BASE_URL
langfuse_handler = CallbackHandler()

helper_llm = ChatQwQ(
    api_key=config.QWEN_API_KEY,
    # model=config.MODEL_SLEEP_AI,
    model="qwen3.7-flash",
    temperature=0.10,
    top_p=0.80,
    extra_body={
        "enable_thinking": False,
    },
    # callbacks=[langfuse_handler],
)


main_llm=ChatQwQ(
    api_key=config.QWEN_API_KEY,
    model=config.MODEL_SLEEP_AI,
    temperature=0.35,
    top_p=0.9001,
    extra_body={
        "enable_thinking": True,
        "thinking_budget": 350,
    },
    # callbacks=[langfuse_handler],
)

classifier_llm = ChatQwQ(
    api_key=config.QWEN_API_KEY,
    # model=config.MODEL_SLEEP_AI,
    model="qwen3.7-flash",
    temperature=0.1,
    extra_body={
        "enable_thinking": False,
        # "thinking_budget": 10,
    },
    # callbacks=[langfuse_handler],
)