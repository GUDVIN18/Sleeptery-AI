from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from app.dialog_ai.resources.schemas.dialog import (
    DialogAi,
    HelperLLMResponse
)


parser_main_llm = JsonOutputParser(pydantic_object=DialogAi)
parser_helper_llm = JsonOutputParser(pydantic_object=HelperLLMResponse)