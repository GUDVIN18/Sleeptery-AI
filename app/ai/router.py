from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import Connection
from typing import List, Dict, Any
from .resources.schemas.sleepai import ResponseSleepAi, UploadSleepAi
from .resources.pipline_sleepai import geration_pipe
# from loguru import logger as log
from app.include.logging_config import logger as log


router = APIRouter()

@router.post(
    "/analyze",
    name="Получить совет от SleepAI",
    response_model=ResponseSleepAi
)
async def analyze_sleep(data: UploadSleepAi):
    log.success(f"Успешно приняли sleep_json!")
    sleepai_answer = await geration_pipe(data.sleep_json)
    return ResponseSleepAi(
        sleep_assessment=sleepai_answer.sleep_assessment,
        response=f"{sleepai_answer.response} {sleepai_answer.diary}",
    )