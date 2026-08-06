from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.engine import Connection
from typing import List, Dict, Any
from .resources.schemas.sleepai import ResponseSleepAi, UploadSleepAi, SleepGraphAi
from .resources.pipline import geration_pipe
# from loguru import logger as log
from app.include.logging_config import logger as log
from ..include.permissions import secret_access
from .resources.redis_async_client import AsyncRedisClient

router = APIRouter()

@router.post(
    "/analyze",
    response_model=ResponseSleepAi,
    dependencies=[Depends(secret_access)],
    name="Получить совет от SleepAI",
)
async def analyze_sleep(data: UploadSleepAi):
    log.info(f"REQUEST: {data}")
    try:
        async with AsyncRedisClient(
            user_id=data.user_id,
            sleep_date=data.sleep_date,
            app_version=data.app_version
        ) as client:
            if await client.create_cache_advice():
                log.success(f"{data.app_version} user_id={data.user_id}: GENERATION ADVICE!")
                sleepai_answer: SleepGraphAi = await geration_pipe(data=data)
                return ResponseSleepAi(
                    sleep_assessment=sleepai_answer.sleep_assessment,
                    response=f"{sleepai_answer.response} {sleepai_answer.diary_recommendation}",
                    mission=sleepai_answer.mission,
                    buttons=sleepai_answer.buttons
                )
            else:
                raise HTTPException(
                    status_code=409, 
                    detail="Совет уже генерируется."
                )
    except Exception as e:
        log.error(f"Ошибка при анализе сна: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при анализе сна")