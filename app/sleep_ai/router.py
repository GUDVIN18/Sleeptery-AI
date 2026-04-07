from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import Connection
from typing import List, Dict, Any
from .resources.schemas.sleepai import ResponseSleepAi, UploadSleepAi, SleepGraphAi
from .resources.pipline import geration_pipe
# from loguru import logger as log
from app.include.logging_config import logger as log
from ..include.permissions import secret_access


router = APIRouter()

@router.post(
    "/analyze",
    response_model=ResponseSleepAi,
    dependencies=[Depends(secret_access)],
    name="Получить совет от SleepAI",
)
async def analyze_sleep(data: UploadSleepAi):
    try:
        log.success(f"Успешно приняли сон!")
        sleepai_answer: SleepGraphAi = await geration_pipe(data.sleep_json)
        return ResponseSleepAi(
            sleep_assessment=sleepai_answer.sleep_assessment,
            response=f"{sleepai_answer.response} {sleepai_answer.diary_recommendation}",
            mission=sleepai_answer.mission,
            buttons=sleepai_answer.buttons
        )
    except Exception as e:
        log.error(f"Ошибка при анализе сна: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при анализе сна")