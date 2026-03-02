import asyncio
import logging
import os
from datetime import timedelta

from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

async def clean_messages():
    db_url = os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("No database URL provided in DB_CONNECTION or DATABASE_URL")
        return

    # In case the db specifies postgres://, SQLAlchemy requires postgresql://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url)
    
    try:
        async with engine.begin() as conn:
            # Delete messages older than 90 days
            query = text("DELETE FROM messages WHERE created_at < NOW() - INTERVAL '90 days'")
            result = await conn.execute(query)
            logger.info(f"Deleted {result.rowcount} obsolete messages.")
    except Exception as e:
        logger.error(f"Failed to clean messages: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(clean_messages())
