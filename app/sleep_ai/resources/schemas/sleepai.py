from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ResponseFormat:
    analysis: List[str]

@dataclass
class ResponseFormatAi(BaseModel):
    reasoning: str = Field(
        description="Скрытый этап. 1) Анализ динамики (лучше/хуже/в норме). 2) Выбор факта из <KNOWLEDGE_BASE> (механизм). 3) Проверка History (не повторять выражения, смысл! И не повторять миссии/ритуалы). 4) Формирование гипотезы."
    )
    sleep_assessment: str = Field(
        description="10-20 токенов. Анализ качества ночи (динамика восстановления и пробуждений). Без сухих цифр."
    )
    response: str = Field(
        description="60-80 токенов. Основной совет. Структура: 'Инсайт из <KNOWLEDGE_BASE> (почему это так)' -> 'Рекомендация (что сделать) из <KNOWLEDGE_BASE>'. Подробная логика описана в system"
    )
    diary_recommendation: str = Field(
        description=(
            "СТРОГАЯ ЛОГИКА ФОРМИРОВАНИЯ: "
            "1) Если дневник пуст: "
            "1.1) при отсутствии действия / миссии / ритуала в 'response' — в конце добавить короткое напоминание: "
            "Заполни дневник сна — это поможет команде Sleeptery точнее подобрать советы. "
            "1.2) при наличии ритуала — мягко предложить внести ЭТОТ ритуал в дневник, "
            "указав, что так Sleeptery сможет проще отслеживать прогресс."
        )
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