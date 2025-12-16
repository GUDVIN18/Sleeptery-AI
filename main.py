from fastapi import Depends, FastAPI
from fastapi.security import APIKeyHeader
from app.router import main_router


app = FastAPI(
    title="Microservice AI",
    version="0.1.0",
    openapi_tags=[{"name": "SleepAI", "description": "Взаимодействие с AI Sleeptery."}],
    dependencies=[
        Depends(APIKeyHeader(name='Secret', scheme_name='api_secret', auto_error=False))
    ],
)
app.include_router(main_router)