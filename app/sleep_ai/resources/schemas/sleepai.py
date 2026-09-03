from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, List
from enum import Enum
from .app import AppVersion
from datetime import date

@dataclass
class ResponseFormat:
    recommendations: List[str] = Field(
        description=(
            "Список из 1 или 2 рекомендаций, выбранных строго из Mapping. "
            "Каждая строка должна полностью совпадать с названием совета (без изменений). "
            "Разрешены только значения из списка. "
            "Нельзя добавлять объяснения, комментарии или любые другие поля. "
            "Если явной проблемы нет — вернуть ['Совет 1: Что такое сон']."
        )
    )

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
    RITUAL_ADVICE='ritual_advice'
    GOAL_ADVICE="goal_advice"
    
class AdviceClassifier(BaseModel):
    advice_type: AdviceType

class UploadSleepAi(BaseModel):
    app_version: Optional[AppVersion] = Field(AppVersion.PROD, description="Версия приложения")
    sleep_date: Optional[date] = Field(None, description="Дата сна в формате YYYY-MM-DD")
    user_id: Optional[int] = Field(None, description="ID пользователя")
    sleep_json: Dict[str, Any] = Field(
        description="Сон пользователя"
    )
    hash_id: Optional[str] = Field(None, description="Хэш совета")

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
    user_goal: Optional[Any] = Field(
        default=None,
        description="Цель пользователя на сегодня"
    )
    user_rituals: Optional[List[Any]] = Field(
        default=None,
        description="Ритуалы пользователя на сегодня"
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
            "Сформируй не более одного напоминания о дневнике сна и только если дневник пуст. "
            "Перед формированием проверь весь текст поля 'response': "
            "если в нём уже есть любое предложение заполнить дневник сна, записать ритуал "
            "или отслеживать прогресс через дневник — верни None. "
            "Если упоминания дневника в 'response' нет: "
            "1) при наличии ритуала предложи записать именно этот ритуал в дневник, "
            "чтобы Sleeptery могла отслеживать прогресс; "
            "2) при отсутствии ритуала, действия или миссии верни одно короткое напоминание: "
            "'Заполни дневник сна — это поможет команде Sleeptery точнее подобрать советы.' "
            "Не повторяй рекомендацию, не создавай несколько вариантов и не добавляй её "
            "одновременно в 'response' и 'diary_recommendation'."
        ),
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