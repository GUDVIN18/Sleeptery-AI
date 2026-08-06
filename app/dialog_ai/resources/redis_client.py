import requests
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_classic.schema import BaseChatMessageHistory
import redis
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.include.logging_config import logger as log
from app.include.config import config


class RedisClient:
    def __init__(
            self,
            session_id: str,
            url:str = "redis://:hUvput-vujfow-ganma3@82.22.184.82:6379/0",
            key_prefix: str = "dialog_ai_history:"
        ):
        self.session_id=session_id
        self.url=url
        self.key_prefix=key_prefix
        self.client=redis.from_url(url, decode_responses=True)
        self.key=f"{key_prefix}{session_id}"
        self.ttl=259200 #3дн

    def get_session_history_v1(
            self, 
        ) -> RedisChatMessageHistory:
        user_id, sleep_date = self.session_id.split("_")
        # Получаем из redis
        history = RedisChatMessageHistory(
            session_id=self.session_id,
            url=self.url,
            key_prefix=self.key_prefix,
            ttl=self.ttl #3дн
        )
        log.info(f"history message {self.session_id=}: {history.messages=}")

        if not history.messages:
            try:
                response = requests.get(
                    f"{config.CHAT_HISTORY_URL}", 
                    params={
                        'user_id': user_id,
                        'sleep_date': sleep_date,
                        'page_size': 100
                    },
                    headers={'Authorization': f'Bearer {config.DOCKER_SECRET}'}
                )
                if response.status_code == 200:
                    data = response.json()
            
                    messages = []
                    for msg in data:
                        if msg['role'] == 'user':
                            messages.append(HumanMessage(content=msg['text']))
                        else:
                            messages.append(AIMessage(content=msg['text']))
                    if messages:
                        history.add_messages(messages)
                else:
                    log.warning(f"API history failed: {response.status_code}")
            except Exception as e:
                log.error(f"Error fetching history from API: {e}")
                history = []
        return history


    def get_session_history_v2(
            self,
            user_id: int,
            sleep_date: str
        ) -> list[BaseChatMessageHistory]:

        history = self.client.lrange(self.key, 0, -1)
        lc_messages = []
        for msg_str in history:
            data = json.loads(msg_str)
            if data['role'] == 'user':
                lc_messages.append(HumanMessage(content=data['content']))
            if data['role'] == 'ai':
                lc_messages.append(AIMessage(content=data['content']))
        if lc_messages is None or len(lc_messages) == 0:
            log.info(f"No messages found in Redis. Go to fetch from API.")
            try:
                response = requests.get(
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
                            self.add_message(
                                role='user' if isinstance(m, HumanMessage) else 'ai',
                                message=m.content
                            )
                else:
                    log.warning(f"API history failed: {response.status_code}")
            except Exception as e:
                log.error(f"Error fetching history from API: {e}")
                lc_messages = []

        return lc_messages



    def add_message(self, role: str, message: str):
        try:
            message_data = json.dumps({
                "role": role,
                "content": message
            }, ensure_ascii=False)
            # Добавляем в конец списка (RPUSH)
            self.client.rpush(self.key, message_data)
        
            self.client.expire(self.key, self.ttl)
            
            log.info(f"Message added to Redis. Key: {self.key}, TTL.")
        except Exception as e:
            log.error(f"Redis add_message error: {e}")
