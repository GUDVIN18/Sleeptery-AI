import datetime as dt
import httpx
from app.include.config import config
from app.include.logging_config import logger as log


class SleepteryDairyAPI:
    _client=httpx.AsyncClient(
        base_url=config.ST_DAIRY_URL,
        timeout=15
    )

    @classmethod
    async def get_user_goal(
            cls,
            user_id: int,
            date: dt.date
    ):
        response = await cls._client.get(
            f"/user-goal/inner/user/{user_id}",
            headers={
                "Authorization": config.DOCKER_SECRET
            },
            params={
                "date": date
            }
        )
        data = response.json()
        log.info(f"\n{data}\n")
        return data
    