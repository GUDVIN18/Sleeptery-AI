from confluent_kafka.aio import AIOConsumer
import asyncio
import json
from app.include.config import config
from app.sleep_ai.resources.schemas.sleepai import ResponseSleepAi, UploadSleepAi, SleepGraphAi
from app.sleep_ai.resources.redis_async_client import AsyncRedisClient
from app.sleep_ai.resources.pipline import geration_pipe
from app.include.logging_config import logger as log


background_task: set[asyncio.Task] = set()


async def run():
    log.info(f"Starting Kafka consumer...")
    c = AIOConsumer({
        'bootstrap.servers': config.KAFKA_BROKER_URL_DEV,
        'group.id': 'sleep_ai_pending_generation',
        'auto.offset.reset': 'earliest',
        'isolation.level': 'read_committed',
        'enable.auto.commit': False
    })
    await c.subscribe(['sleep_ai_pending_generation'])
    try:
        while True:
            msg = await c.poll(1.0)

            if msg is None:
                continue
            if msg.error():
                print("Consumer error: {}".format(msg.error()))
                continue

            # print('Received message: {}'.format(msg.value().decode('utf-8'))) # Получаем
            log.info(f"Сообщение получено от: user_id={msg.key().decode('utf-8')}")
            try:
                upload_data = json.loads(
                    msg.value().decode('utf-8')
                )
                data = UploadSleepAi(
                    user_id=upload_data['user_id'],
                    sleep_date=upload_data['sleep_date'],
                    app_version=upload_data['app_version'],
                    sleep_json=upload_data['sleep_json']
                )
                try:
                    async with AsyncRedisClient(
                        user_id=data.user_id,
                        sleep_date=data.sleep_date,
                        app_version=data.app_version
                    ) as client:
                        if await client.create_cache_advice():
                            # sleepai_answer: SleepGraphAi = await geration_pipe(data=data)
                            task = asyncio.create_task(
                                name=f"background_generation_user_{data.user_id}",
                                coro=geration_pipe(data),
                            )
                            task.add_done_callback(background_task.discard)
                            background_task.add(task)
                            log.success(f"{data.app_version} user_id={data.user_id}: GENERATION ADVICE STARTED!")
                            await c.commit()
                        else:
                            log.error(f"user_id={data.user_id}: ADVICE ALREADY GENERATING!")
                            raise Exception("Совет уже генерируется.")
                except Exception as e:
                    log.error(f"Ошибка при анализе сна: {e}")
                    raise Exception("Ошибка при анализе сна")   
            except Exception as e:
                print(f"Error decoding message key: {e}")
    finally:
        await c.unsubscribe()
        await c.close()


if __name__ == "__main__":
    asyncio.run(run())
