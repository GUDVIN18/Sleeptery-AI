from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from app.sleep_ai.resources.schemas.sleepai import (
    AdviceLLMResponse,
    AdviceClassifier,
)


# parser = JsonOutputParser(pydantic_object=SleepGraphAi)
parser = JsonOutputParser(pydantic_object=AdviceLLMResponse)
classifier_parser = JsonOutputParser(
    pydantic_object=AdviceClassifier
)

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