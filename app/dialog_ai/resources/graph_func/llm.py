import os
from pathlib import Path
from langchain_core.globals import set_debug
from app.include.config import config
from langchain_qwq import ChatQwQ
from app.include.logging_config import logger as log
from langfuse.langchain import CallbackHandler


BASE_DIR = Path(__file__).resolve().parent.parent.parent
set_debug(False)
try:
    SYSTEM_INSTRUCTION = (BASE_DIR / "context" / "2025-12-12-instruction.txt").read_text(encoding="utf-8")
    CONTEXT_PROMPT = (BASE_DIR / "context" / "2026-02-20-contextualize_prompt.txt").read_text(encoding="utf-8")
except Exception as e:
    log.error(f"Failed to load prompts: {e}")

os.environ["LANGFUSE_PUBLIC_KEY"] = config.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = config.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = config.LANGFUSE_BASE_URL

langfuse_handler = CallbackHandler()


llm_analytics = ChatQwQ(
    api_key=config.QWEN_API_KEY,
    model=config.MODEL_DIALOG_AI,
    temperature=0.1,
    top_p=0.95,
    extra_body={
        "enable_thinking": True,
        "thinking_budget": 40,
    },
    callbacks=[langfuse_handler],
)
main_llm=ChatQwQ(
    api_key=config.QWEN_API_KEY,
    model=config.MODEL_DIALOG_AI,
    temperature=0.3,
    top_p=0.95,
    extra_body={
        "enable_thinking": True,
        "thinking_budget": 110,
    },
    callbacks=[langfuse_handler],
)