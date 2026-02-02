from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict, Optional
import datetime as dt


# Для пайплайна
@dataclass
class ResponseFormatAi(BaseModel):
    answer: str = Field(
        description="Ответ AI" # Который мы требуем от модели
    )


# для fastapi роутера
class ResponseDialogAi(BaseModel):
    answer: str = Field(
        description="Ответ AI" # Который мы возвращаем пользователю
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


class ResponseMessage(BaseModel):
    type: str
    message: str
