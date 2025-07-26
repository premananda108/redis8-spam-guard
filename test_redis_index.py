
import asyncio
import aioredis
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_index():
    """Подключается к Redis и выполняет FT.INFO"""
    redis_client = None
    try:
        logger.info("Connecting to Redis...")
        redis_client = await aioredis.from_url("redis://localhost:6379", decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection successful.")

        logger.info("Fetching info for index 'post_vectors'...")
        index_info = await redis_client.execute_command("FT.INFO", "post_vectors")
        
        logger.info("--- FT.INFO post_vectors ---")
        # Выводим информацию в читаемом виде
        info_dict = {index_info[i]: index_info[i+1] for i in range(0, len(index_info), 2)}
        for key, value in info_dict.items():
            print(f"{key}: {value}")
        logger.info("--------------------------")

    except aioredis.exceptions.ResponseError as e:
        logger.error(f"Redis command failed: {e}")
        logger.error("This likely means the index 'post_vectors' does not exist.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed.")

if __name__ == "__main__":
    asyncio.run(check_index())
