from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from app.dialog_ai.resources.schemas.dialog import (
    DialogAi,
)


parser = JsonOutputParser(pydantic_object=DialogAi)