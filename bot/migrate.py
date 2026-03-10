import asyncio
import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_migration():
    """Run database migration within the Railway environment."""
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD')
    host = os.getenv('POSTGRES_HOST', 'pgvector.railway.internal')
    db = os.getenv('POSTGRES_DB', 'railway')
    
    if not password:
        logger.error("POSTGRES_PASSWORD environment variable is not set!")
        return

    # Construct SQLAlchemy async URL
    # We use pgvector.railway.internal for the host as it's the internal service name
    url = f"postgresql+asyncpg://{user}:{password}@{host}/{db}"
    logger.info(f"Connecting to database at {host}...")
    
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            logger.info("Running migrations...")
            await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS english_level TEXT;"))
            await conn.execute(text("ALTER TABLE pipeline_logs ADD COLUMN IF NOT EXISTS sources JSONB;"))
            await conn.execute(text("ALTER TABLE pipeline_logs ADD COLUMN IF NOT EXISTS error TEXT;"))
            logger.info("✅ Migrations successful!")
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("✅ Columns already exist.")
        else:
            logger.error(f"❌ Migration failed: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(run_migration())
