from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import Connection
from typing import List, Dict, Any
from .resources.schemas.sleepai import ResponseSleepAi, UploadSleepAi
from .resources.pipline_sleepai import geration_pipe
from loguru import logger as log


router = APIRouter()

@router.post(
    "/analyze",
    name="Получить совет от SleepAI",
    response_model=ResponseSleepAi
)
async def analyze_sleep(sleep_json=UploadSleepAi):
    
    sleepai_answer = await geration_pipe(sleep_json)
    return ResponseSleepAi(
        sleep_assessment=sleepai_answer.sleep_assessment,
        response=f"{sleepai_answer.response} {sleepai_answer.diary}",
    )