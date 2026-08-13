import httpx
import json
import redis.asyncio as redis  # Используем асинхронный клиент Redis
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .schemas import AppVersion
from app.include.logging_config import logger as log
from app.include.config import config


class AsyncRedisClient:
    def __init__(
            self,
            user_id: int,
            sleep_date: str,
            app_version: AppVersion,
            url: str = f"redis://:{config.REDIS_PASS}@redis:6379/0",
            key_prefix: str = "generate_advice:"
        ):
        self.url = url
        self.key_prefix = key_prefix
        # Создаем асинхронный пул соединений с Redis
        self.client = redis.from_url(url, decode_responses=True)
        self.key = f"{key_prefix}{app_version}_{user_id}_{sleep_date}"
        self.ttl = 360 # 6 минут

    # вход в контекстный менеджер
    async def __aenter__(self):
        return self
    # автоматическое закрытие соединения
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def create_cache_advice(self) -> bool:
        try:
            result = await self.client.set(
                self.key,
                f"ключ совета: {self.key}",              
                ex=self.ttl,
                nx=True           
            )

            if result:
                log.debug(f"Ключ {self.key} успешно создан (TTL={self.ttl})")
                return True
            else:
                log.debug(f"Ключ {self.key} уже существует")
                return False

        except Exception as e:
            log.error(f"Ошибка Redis: {e}")
            return False