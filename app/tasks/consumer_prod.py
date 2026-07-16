from confluent_kafka import Consumer
import asyncio
import json

from app.include.config import config
from app.sleep_ai.resources.schemas.sleepai import UploadSleepAi
from app.sleep_ai.resources.redis_async_client import AsyncRedisClient
from app.sleep_ai.resources.pipline import geration_pipe
from app.include.logging_config import logger as log


async def run():
    log.info("Starting Kafka consumer...")

    consumer = Consumer({
        "bootstrap.servers": config.KAFKA_BROKER_URL_PROD,
        "group.id": "sleep_ai_pending_generation",
        "auto.offset.reset": "earliest",
        "isolation.level": "read_committed",
        "enable.auto.commit": False,
        "max.poll.interval.ms": 600_000,
    })

    consumer.subscribe(["sleep_ai_pending_generation"])

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                log.error(f"Consumer error: {msg.error()}")
                continue

            try:
                user_id = (
                    msg.key().decode("utf-8")
                    if msg.key() is not None
                    else "unknown"
                )

                log.info(f"Сообщение получено от: user_id={user_id}")

                upload_data = json.loads(
                    msg.value().decode("utf-8")
                )

                data = UploadSleepAi(
                    user_id=upload_data["user_id"],
                    sleep_date=upload_data["sleep_date"],
                    app_version=upload_data["app_version"],
                    sleep_json=upload_data["sleep_json"],
                )

                async with AsyncRedisClient(
                    user_id=data.user_id,
                    sleep_date=data.sleep_date,
                    app_version=data.app_version,
                ) as client:
                    cache_created = await client.create_cache_advice()

                if not cache_created:
                    log.error(
                        f"user_id={data.user_id}: "
                        f"ADVICE ALREADY GENERATING!"
                    )

                    # Сообщение обработано, повторная генерация не нужна
                    consumer.commit(message=msg, asynchronous=False)
                    continue

                log.success(
                    f"{data.app_version} "
                    f"user_id={data.user_id}: "
                    f"GENERATION ADVICE STARTED!"
                )

                await geration_pipe(data)

                consumer.commit(message=msg, asynchronous=False)

                log.success(
                    f"{data.app_version} "
                    f"user_id={data.user_id}: "
                    f"GENERATION ADVICE FINISHED!"
                )

            except json.JSONDecodeError:
                log.exception("Ошибка декодирования Kafka-сообщения")

            except Exception:
                log.exception("Ошибка при обработке сообщения сна")

    except KeyboardInterrupt:
        log.info("Kafka consumer stopped")

    finally:
        consumer.unsubscribe()
        consumer.close()
        log.info("Kafka consumer closed")


if __name__ == "__main__":
    asyncio.run(run())