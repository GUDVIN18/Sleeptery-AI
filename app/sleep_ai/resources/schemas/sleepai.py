from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, List
from enum import Enum
from .app import AppVersion
from datetime import date

@dataclass
class ResponseFormat:
    analysis: List[str]

# для fastapi роутера
class ResponseSleepAi(BaseModel):
    sleep_assessment: str = Field(description="Анализ")
    response: str = Field(description="Совет")

    mission: Optional[str] = Field(
        None,
        description="Миссия"
    )

    buttons: Optional[str] = Field(
        None,
        description="Кнопка с добавлением совета"
    )

class AdviceType(str, Enum):
    GENERATION_ADVICE='generation_advice'
    ANALYSIS_ADVICE='analysis_advice'
    
class AdviceClassifier(BaseModel):
    advice_type: AdviceType

class UploadSleepAi(BaseModel):
    app_version: Optional[AppVersion] = Field(AppVersion.PROD, description="Версия приложения")
    sleep_date: Optional[date] = Field(None, description="Дата сна в формате YYYY-MM-DD")
    user_id: Optional[int] = Field(None, description="ID пользователя")
    sleep_json: Dict[str, Any] = Field(
        description="Сон пользователя"
    )

class SleepGraphAi(UploadSleepAi):
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
        description=(
            "Заполни ТОЛЬКО если в поле 'response' содержит конкретное действие-ритуал, "
            "которое пользователь может выполнить (утренний/вечерний ритуал, упражнение, привычка и т.п.). "
            "Запиши краткое название этого действия в 2-5 слов — точно отражающее суть из 'response'. "
            "Если конкретного действия нет — оставь null."
        )
    )
    buttons: Optional[List[str]] = Field(default=None)