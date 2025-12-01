from fastapi import FastAPI
from app.router import main_router


app = FastAPI(
    title="Microservice AI",
    version="0.1.0",
    openapi_tags=[{"name": "SleepAI", "description": "Взаимодействие с AI Sleeptery."}],
)
app.include_router(main_router)