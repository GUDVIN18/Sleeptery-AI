import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import Connection
from typing import List, Dict, Any
from .resources.schemas.dialog import (
    ResponseDialogAi, 
    UploadDialogAi, 
    ResponseMessage,
    DialogAi
)
from .resources.pipline import geration_pipe
from .resources.redis_client import RedisClient
# from loguru import logger as log
from app.include.logging_config import logger as log
from ..include.permissions import secret_access
from .resources.exceptions import DialogAiErrorGeneration


router = APIRouter()

@router.post(
    "/chat",
    response_model=ResponseDialogAi,
    dependencies=[Depends(secret_access)],
    name="Задать вопрос и получить ответ",
)
async def dialog(
    data: UploadDialogAi,
):
    log.info(f"{data=}")
    try:
        dialogai_answer: DialogAi = await geration_pipe(data=data)
        return ResponseDialogAi(
            message=dialogai_answer.answer,
            buttons=dialogai_answer.buttons
        )
    except Exception as e:
        log.error(f"Unhandled error in /chat endpoint: {e}")
        raise DialogAiErrorGeneration

@router.get(
    "/history",
    response_model=List[ResponseMessage],
    dependencies=[Depends(secret_access)],
    name="Получить историю диалога",
)
async def get_dialog_history(
    user_id: int = Query(
        description="Уникальный идентификатор пользователя"
    ),
    sleep_date: dt.date = Query(
        description="Дата сна"
    )
):
    log.info(f"Запрос истории диалога для пользователя {user_id} и даты сна {sleep_date}")
    get_all_message = RedisClient(
        session_id=f"{user_id}_{sleep_date}"
    ).get_session_history().messages
    result: List[ResponseMessage] = []
    for msg in get_all_message:
        result.append(
            ResponseMessage(
                type=msg.type,
                message=msg.content
            )
        )
    return result