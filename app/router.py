from fastapi import APIRouter
from .sleep_ai.router import router as incidents_router
from .dialog_ai.router import router as dialogai_router

main_router = APIRouter()


main_router.include_router(
    incidents_router,
    tags=["SleepAI Pipline"],
    prefix='/ai'
)

main_router.include_router(
    dialogai_router,
    tags=["DialogAI Pipline"],
    prefix='/dialog_ai'
)
