from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import Connection
from typing import List, Dict, Any
from .resources.schemas.dialog import ResponseDialogAi, UploadDialogAi
from .resources.pipline import geration_pipe
# from loguru import logger as log
from app.include.logging_config import logger as log
from ..include.permissions import secret_access


router = APIRouter()

@router.post(
    "/dialog",
    response_model=ResponseDialogAi,
    dependencies=[Depends(secret_access)],
    name="Задать вопрос и получить ответ",
)
async def dialog(
    data: UploadDialogAi,
):
    dialogai_answer = await geration_pipe(data=data)
    return ResponseDialogAi(
        answer=dialogai_answer.answer
    )

@router.get(
    "/history",
    response_model=None,
    dependencies=[Depends(secret_access)],
    name="Получить историю диалога",
)
async def get_dialog_history():
    pass