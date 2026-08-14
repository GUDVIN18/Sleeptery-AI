from app.include.logging_config import logger as log
from fastapi import Depends, FastAPI
from fastapi.security import APIKeyHeader
from app.router import main_router
import uvicorn


log.success("Starting Microservice AI...")
app = FastAPI(
    title="Microservice AI",
    version="0.1.0",
    openapi_tags=[{"name": "SleepAI", "description": "Взаимодействие с AI Sleeptery."}],
    dependencies=[
        Depends(APIKeyHeader(name='Secret', scheme_name='api_secret', auto_error=False))
    ],
)
app.include_router(main_router)

if __name__ == "__main__":
    log.info("Starting debug uvicorn")
    uvicorn.run(
        "app.main:app",
        host='0.0.0.0',
        port=8882,
        reload=True,
        workers=1,
        log_level='debug',
    )
    log.info("Uvicorn stopped")
