from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict
import datetime as dt


# Для пайплайна
@dataclass
class ResponseFormatAi(BaseModel):
    answer: str = Field(
        description="Ответ AI"
    )
    # history_dialog = Field(
    #     description="История диалога"
    # )


# для fastapi роутера
class ResponseDialogAi(BaseModel):
    answer: str = Field(
        description="Ответ AI"
    )

class UploadDialogAi(BaseModel):
    question: str = Field(
        description="Вопрос пользователя"
    )
    user_id: int = Field(
        description="Уникальный идентификатор пользователя"
    )
    sleep_data: dt.date = Field(
        description="Дата сна"
    )