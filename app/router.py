from fastapi import APIRouter
from .ai.router import router as incidents_router

main_router = APIRouter()


main_router.include_router(
    incidents_router,
    tags=["Pipline"],
    prefix='/ai'
)
