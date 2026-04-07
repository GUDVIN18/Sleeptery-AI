from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, List
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
import datetime as dt
from enum import Enum


class ButtonType(str, Enum):
    ADD_HABIT = "add_habit"
    # SUGGEST_RITUAL = "suggest_ritual"

class Button(BaseModel):
    type: ButtonType = Field(
        description=(
            "Тип кнопки: "
            # f"'{ButtonType.ADD_TO_DIARY.value}' — добавить конкретный совет/ритуал из ответа в дневник; "
            f"'{ButtonType.ADD_HABIT.value}' — предложить пользователю ритуал/совет для отслеживания и добавления в дневник."
        )
    )
    title: str = Field(
        description=(
            "название кнопки. "
            # "Для 'add_to_diary': 'Добавить [название совета] в дневник'. "
            f"Для '{ButtonType.ADD_HABIT.value}': краткое название ритуала, например 'Плотные шторы', 'Проветрить', 'Выключить свет' и т.д."
        )
    )
    text: str = Field(
        description=(
            "Текст привычки для добавления в дневник. Без формулировок типа 'Добавить [название] в дневник', только название"
        )
    )


# для fastapi роутера
class ResponseDialogAi(BaseModel):
    message: str = Field(description="Ответ AI")
    buttons: Optional[List[Button]] = Field(
        None,
        description="Кнопки"
    )


class UploadDialogAi(BaseModel):
    message: str = Field(
        description="Вопрос пользователя"
    )
    user_id: int = Field(
        description="Уникальный идентификатор пользователя"
    )
    sleep_date: dt.date = Field(
        description="Дата сна"
    )
    sleep_json: Optional[Dict[str, Any]] = Field(
        None,
        description="Данные сна пользователя"
    )
    sleep_assessment: Optional[str] = Field(
        None,
        description="Совет по улучшению сна от SleepAI"
    )


class DialogAi(UploadDialogAi):
    test_mode: bool = Field(False)
    context_vector_db: str = Field(
        None,
        description="Контекст из векторноый базы знаний"
    )
    history_messages: List[BaseMessage] = Field(
        None,
        description="История сообщений пользователя"
    )
    context_rag_search: str = Field(
        None,
        description="Контекст для поиска в векторной БД"
    )
    answer: str = Field(
        None,
        description=(
            "Ответ пользователю. "
            "Если будут сгенерированы кнопки (buttons) — в самом конце ответа добавь строку: "
            "'Добавь в свой дневник следующие ритуалы:'. "
            "Если кнопок нет — эту фразу не добавляй."
        )
    )

    buttons: Optional[List[Button]] = Field(
        default=None,
        description=f"""
    Кнопки для UI. Генерируй ТОЛЬКО при необходимости на основе текущего answer и context_vector_db.

    ПРАВИЛА:
    1. {ButtonType.ADD_HABIT.value} (1-3 шт): конкретные действия/предметы из answer
    - Извлекай из текста ответа, не придумывай
    - Формат: краткое существительное/действие, 1-3 слова
    - Пример логики: если в answer упомянуты "шторы блэкаут" и "маска" → label = "Шторы блэкаут", "Маска для сна"

    ЗАПРЕЩЕНО:
    - Копировать кнопки из примеров обучения
    - Генерировать кнопки не связанные с текущим answer
    - Генерировать если ответ — уточняющий вопрос
    """)
    # 2. add_to_diary (всегда 1 шт): общая тема текущего совета
    # - Формат: "Добавить [тема] в дневник"
    # - Тему бери из сути вопроса пользователя, не из примеров

class ResponseMessage(BaseModel):
    type: str
    message: str
