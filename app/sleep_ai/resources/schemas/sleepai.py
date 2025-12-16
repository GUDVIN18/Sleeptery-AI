from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ResponseFormat:
    analysis: List[str]

@dataclass
class ResponseFormatAi(BaseModel):
    reasoning: str = Field(
        description="Скрытый этап. 1) Анализ динамики (лучше/хуже). 2) Выбор факта из <RAG_CONTEXT> (механизм). 3) Проверка History (не повторяться! И не повторять миссии/ритуалы). 4) Формирование гипотезы."
    )
    sleep_assessment: str = Field(
        description="10-20 токенов. Анализ качества ночи (динамика восстановления и пробуждений). Без сухих цифр."
    )
    response: str = Field(
        description="60-80 токенов. Основной совет. Структура: 'Инсайт из <RAG_CONTEXT> (почему это так)' -> 'Рекомендация (что сделать) из <RAG_CONTEXT>'."
    )
    diary_recommendation: str = Field(
        description="СТРОГАЯ ЛОГИКА: Если в поле 'response' ты предложил конкретное действие/миссию — призови записать ИМЕННО ЭТО в дневник. Если совет теоретический — просто напомни заполнить дневник."
    )
    mission: str = Field(
        description="Если в поле 'response' ты предложил конкретное действие/миссию — запиши в это поле только ее название! Иначе оставь пустым."
    )

class ResponseSleepAi(BaseModel):
    sleep_assessment: str = Field(
        description="Анализ"
    )
    response: str = Field(
        description="Совет"
    )
    mission: str = Field(
        description="Миссия"
    )

class UploadSleepAi(BaseModel):
    sleep_json: Dict[str, Any]