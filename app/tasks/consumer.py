from confluent_kafka import Consumer
import asyncio
import json
from st_bases.telegram import TgLog
from app.include.config import config
from app.sleep_ai.resources.schemas.sleepai import UploadSleepAi
from app.sleep_ai.resources.redis_async_client import AsyncRedisClient
from app.sleep_ai.resources.pipline import geration_pipe
from app.include.logging_config import logger as log


consumer = Consumer({
    "bootstrap.servers": config.KAFKA_BROKER_URL,
    "group.id": "sleep_ai_pending_generation",
    "auto.offset.reset": "earliest",
    "isolation.level": "read_committed",
    "enable.auto.commit": False,
    "max.poll.interval.ms": 600_000,
})

async def run():
    log.info("Starting Kafka consumer...")

    consumer.subscribe(["sleep_ai_pending_generation"])

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                log.error(f"Consumer error: {msg.error()}")
                continue

            log.info(f"Сообщение получено от: user_id={msg.key().decode()}")

            try:
                upload_data = json.loads(msg.value().decode())

                data = UploadSleepAi(
                    user_id=upload_data["user_id"],
                    sleep_date=upload_data["sleep_date"],
                    app_version=upload_data["app_version"],
                    sleep_json=upload_data["sleep_json"],
                    hash_id=upload_data["hash_id"],
                )

                async with AsyncRedisClient(
                    user_id=data.user_id,
                    sleep_date=data.sleep_date,
                    app_version=data.app_version,
                ) as client:
                    if await client.create_cache_advice():
                        result = await geration_pipe(data)

                        if result is not None:
                            log.success(
                                f"{data.app_version} user_id={data.user_id}: GENERATION ADVICE FINISHED!"
                            )
                    else:
                        log.error(
                            f"user_id={data.user_id}: ADVICE ALREADY GENERATING!"
                        )

            except Exception as e:
                log.exception("Ошибка при обработке сообщения")
                await TgLog.error(f"Ошибка при обработке сообщения на ИИ сервере в consumer: \n {e}")
            finally:
                consumer.commit(
                    message=msg,
                    asynchronous=False,
                )
    except Exception as e:
        await TgLog.error(f"Ошибка на ИИ сервере в consumer: \n {e}")
    finally:
        consumer.unsubscribe()
        consumer.close()


if __name__ == "__main__":
    asyncio.run(run())