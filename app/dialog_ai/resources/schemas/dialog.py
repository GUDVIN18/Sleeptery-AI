from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ResponseFormat:
    analysis: list

@dataclass
class ResponseFormatAi(BaseModel):
    reasoning: str = Field(
        description="Скрытый этап. 1) Анализ динамики (лучше/хуже). 2) Выбор факта из RAG (механизм). 3) Проверка History (не повторяться!). 4) Формирование гипотезы."
    )
    sleep_assessment: str = Field(
        description="10-20 токенов. Анализ качества ночи (динамика восстановления и пробуждений). Без сухих цифр."
    )
    response: str = Field(
        description="60-80 токенов. Основной совет. Структура: 'Инсайт из RAG (почему это так)' -> 'Мягкая рекомендация (что сделать)'."
    )
    diary: str = Field(
        description="СТРОГАЯ ЛОГИКА: Если в поле 'response' ты предложил конкретное действие/миссию — призови записать ИМЕННО ЭТО в дневник. Если совет теоретический — просто напомни заполнить дневник."
    )

class ResponseSleepAi(BaseModel):
    sleep_assessment: str = Field(
        description="Анализ"
    )
    response: str = Field(
        description="Совет"
    )

class UploadSleepAi(BaseModel):
    sleep_json: Dict[str, Any]