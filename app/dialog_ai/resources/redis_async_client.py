import httpx
import json
import redis.asyncio as redis  # Используем асинхронный клиент Redis
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.include.logging_config import logger as log
from app.include.config import config


class AsyncRedisClient:
    def __init__(
            self,
            session_id: str,
            url: str = "redis://:hUvput-vujfow-ganma3@82.22.184.82:6379/0",
            key_prefix: str = "dialog_ai_history:"
        ):
        self.session_id = session_id
        self.url = url
        self.key_prefix = key_prefix
        # Создаем асинхронный пул соединений с Redis
        self.client = redis.from_url(url, decode_responses=True)
        self.key = f"{key_prefix}{session_id}"
        self.ttl = 259200  # 3 дня

    # вход в контекстный менеджер
    async def __aenter__(self):
        return self
    # автоматическое закрытие соединения
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def get_session_history_v2(
            self,
            user_id: int,
            sleep_date: str
        ) -> list:
        
        # Асинхронный запрос к Redis
        history = await self.client.lrange(self.key, 0, -1)
        lc_messages = []
        
        for msg_str in history:
            data = json.loads(msg_str)
            if data['role'] == 'user':
                lc_messages.append(HumanMessage(content=data['content']))
            elif data['role'] == 'ai':
                lc_messages.append(AIMessage(content=data['content']))
                
        if not lc_messages:
            log.info("No messages found in Redis. Go to fetch from API.")
            try:
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.get(
                        f"{config.CHAT_HISTORY_URL}", 
                        params={
                            'user_id': user_id,
                            'sleep_date': sleep_date,
                            'page_size': 100
                        },
                        headers={'Authorization': f'Bearer {config.DOCKER_SECRET}'}
                    )
                    
                log.info(f"Response from history API: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    for msg in data:
                        text = (
                            msg.get("message", {})
                            .get("payload", {})
                            .get("message")
                        )
                        if msg.get("author") == 'user':
                            lc_messages.append(HumanMessage(content=text))
                        else:
                            lc_messages.append(AIMessage(content=text))
                            
                    if lc_messages:
                        for m in lc_messages:
                            # Асинхронное сохранение в Redis
                            await self.add_message(
                                role='user' if isinstance(m, HumanMessage) else 'ai',
                                message=m.content
                            )
                else:
                    log.warning(f"API history failed: {response.status_code}")
            except Exception as e:
                log.error(f"Error fetching history from API: {e}")
                lc_messages = []

        return lc_messages

    async def add_message(self, role: str, message: str):
        try:
            message_data = json.dumps({
                "role": role,
                "content": message
            }, ensure_ascii=False)
            
            await self.client.rpush(self.key, message_data)
            await self.client.expire(self.key, self.ttl)
            
            log.info(f"Message added to Redis. Key: {self.key}, TTL.")
        except Exception as e:
            log.error(f"Redis add_message error: {e}")