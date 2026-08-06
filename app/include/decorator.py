import time
from .logging_config import logger as log


def current_time(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        log.info(f"Время выполнения функции {func.__name__}: {end_time - start_time:.2f} секунд")
        return result
    return wrapper