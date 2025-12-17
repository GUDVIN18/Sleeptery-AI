from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_classic.schema import BaseChatMessageHistory


class RedisClient:
    def __init__(
            self,
            session_id: str,
            url: str = "redis://localhost:6379/0",
            key_prefix: str = "dialog_ai_history:"
        ):
        self.session_id = session_id
        self.url = url
        self.key_prefix = key_prefix

    def get_session_history(self,) -> BaseChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=self.session_id,
            url=self.url,
            key_prefix=self.key_prefix
        )