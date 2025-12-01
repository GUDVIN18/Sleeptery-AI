from loguru import logger
import sys
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

logger.remove()  # убираем дефолтный логгер Loguru
logger.add(sys.stdout, format="{time} {level} {message}", level=LOG_LEVEL)

# при необходимости файл логов внутри контейнера
# logger.add("/app/logs/app.log", rotation="10 MB", level=LOG_LEVEL)