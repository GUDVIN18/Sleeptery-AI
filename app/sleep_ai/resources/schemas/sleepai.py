from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, List
from enum import Enum


@dataclass
class ResponseFormat:
    analysis: List[str]

@dataclass
class ResponseFormatAi(BaseModel):
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

    button: Optional[str] = Field(
        default=None,
        description=(
            """Кнопка"""
        )
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
    button: Optional[str] = Field(
        None,
        description="Кнопка с добавлением совета"
    )


class UploadSleepAi(BaseModel):
    sleep_json: Dict[str, Any]

class AdviceType(str, Enum):
    GENERATION_ADVICE='generation_advice'
    ANALYSIS_ADVICE='analysis_advice'
    
class AdviceClassifier(BaseModel):
    advice_type: AdviceType

class SleepGraphAi(BaseModel):
    sleep_data: Dict[str, Any]
    user_diary_records: Optional[Any] = Field(
        default=None,
        description="Дневник пользователя за сегодняшний день"
    )

    sleep_daily_stats: Optional[Any] = Field(
        default=None,
        description="Сформированный сон за сегодня"
    )

    sleep_weekly_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Сформированный сон за последние несолько дней"
    )

    history_sleep_assessment: Optional[List[Any]] = Field(
        default=None,
        description="История советов по улучшению сна (как раз они не должны повторяться)"
    )
    context_vector_db: Optional[Any] = Field(
        None,
        description="Контекст из векторноый базы знаний"
    )
    advice_type: Optional[AdviceType] = Field(
        None,
        description="Тип генерируемого совета"
    )
    sleep_assessment: Optional[str] = Field(
        None,
        description="10-20 токенов. Анализ качества ночи (динамика восстановления и пробуждений). Без сухих цифр."
    )
    response: Optional[str] = Field(
        None,
        description="60-80 токенов. Основной совет. Структура: 'Инсайт из <KNOWLEDGE_BASE> (почему это так)' -> 'Рекомендация (что сделать) из <KNOWLEDGE_BASE>'. Подробная логика описана в system"
    )
    diary_recommendation: Optional[str] = Field(
        None,
        description=(
            "СТРОГАЯ ЛОГИКА ФОРМИРОВАНИЯ: "
            "1) Если дневник пуст: "
            "1.1) при отсутствии действия / миссии / ритуала в 'response' — в конце добавить короткое напоминание: "
            "Заполни дневник сна — это поможет команде Sleeptery точнее подобрать советы. "
            "1.2) при наличии ритуала — мягко предложить внести ЭТОТ ритуал в дневник, "
            "указав, что так Sleeptery сможет проще отслеживать прогресс."
        )
    )
    mission: Optional[str] = Field(
        None,
        description="Если в поле 'response' ты предложил конкретное действие/миссию — запиши в это поле только ее название! Иначе оставь пустым."
    )
    button: Optional[Literal["Добавить совет в дневник"]] = Field(
        default=None,
        description=(
            """Если поле mission заполененно, верни
            "Добавить совет в дневник". 
            Если mission нет — верни null."""
        )
    )




class AdviceLLMResponse(BaseModel):
    sleep_assessment: str = Field(
        description="10-20 токенов. Анализ качества ночи. Без сухих цифр."
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
    mission: Optional[str] = Field(
        default=None,
        description="Если в поле 'response' ты предложил конкретное действие/миссию — запиши в это поле только ее название! Иначе оставь пустым."
    )

    button: Optional[str] = Field(
        default=None,
        description=(
            """Кнопка"""
        )
    )