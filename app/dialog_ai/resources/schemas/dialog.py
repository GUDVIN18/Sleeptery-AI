from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, List
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
import datetime as dt
from enum import Enum


class Buttons(BaseModel):
    type: Literal["addAdvice"]
    message: str

# TODO: OLD, Для пайплайна
@dataclass
class ResponseFormatAi(BaseModel):
    answer: str = Field(
        description="Ответ AI" # Который мы возвращаем пользователю
    )
    button: Optional[Buttons] = Field(
        default=None,
        description='''Если в своем ответе ты используешь совет по улучшению сна(ритуал/совет) то выведи кнопку "Добавить совет в дневник". Если в ответе нет вышесказанного - ничего не доабвляй'''
    )


# для fastapi роутера
class ResponseDialogAi(BaseModel):
    answer: str = Field(description="Ответ AI")
    button: Optional[Buttons] = Field(
        default=None,
        description='''Если в своем ответе ты используешь совет по улучшению сна(ритуал/совет) то выведи кнопку "Добавить совет в дневник". Если в ответе нет вышесказанного - ничего не доабвляй'''
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
        description="Ответ пользователю"
    )

    button: Optional[Literal["Добавить совет в дневник"]] = Field(
        default=None,
        description=(
            'Если в ответе есть совет по улучшению сна, верни '
            '"Добавить совет в дневник". '
            'Если совета нет — верни null.'
        )
    )

class ResponseMessage(BaseModel):
    type: str
    message: str
