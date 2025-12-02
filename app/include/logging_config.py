from loguru import logger
import sys
import os
from include.config import config

logger.remove()  # убираем дефолтный логгер Loguru
logger.add(sys.stdout, format="{time} {level} {message}", level=config.LOG_LEVEL)

# при необходимости файл логов внутри контейнера
# logger.add("/app/logs/app.log", rotation="10 MB", level=LOG_LEVEL)