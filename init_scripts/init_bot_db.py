import os
import asyncio
import bot.env

import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig, ListConfig
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from bot.db import Admin, Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_memory_columns(engine):
    """Add new columns and tables if they don't exist."""
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS journey_state JSONB"
            ))
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS conversation_summary TEXT"
            ))
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS english_level TEXT"
            ))
            # Feedback table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tg_id BIGINT REFERENCES users(tg_id) ON DELETE CASCADE,
                    message_id BIGINT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_feedback_tg_id ON feedback(tg_id)"
            ))
            # Pipeline logs table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS pipeline_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tg_id BIGINT NOT NULL,
                    question TEXT NOT NULL,
                    rewritten_question TEXT,
                    cache_hit BOOLEAN DEFAULT FALSE,
                    docs_retrieved INTEGER DEFAULT 0,
                    docs_after_grading INTEGER DEFAULT 0,
                    t_rewrite DOUBLE PRECISION,
                    t_retrieve DOUBLE PRECISION,
                    t_grade DOUBLE PRECISION,
                    t_generate DOUBLE PRECISION,
                    t_total DOUBLE PRECISION NOT NULL,
                    sources JSONB,
                    error TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_pipeline_logs_created "
                "ON pipeline_logs(created_at)"
            ))
            # Add sources and error columns if they don't exist (for existing tables)
            conn.execute(text(
                "ALTER TABLE pipeline_logs ADD COLUMN IF NOT EXISTS sources JSONB"
            ))
            conn.execute(text(
                "ALTER TABLE pipeline_logs ADD COLUMN IF NOT EXISTS error TEXT"
            ))

            # User entities table (structured memory)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_entities (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tg_id BIGINT REFERENCES users(tg_id) ON DELETE CASCADE,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_value TEXT NOT NULL,
                    confidence DOUBLE PRECISION DEFAULT 1.0,
                    source VARCHAR(50) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_user_entities_tg_id ON user_entities(tg_id)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_user_entities_type ON user_entities(entity_type)"
            ))

            # A/B test logs table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ab_test_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tg_id BIGINT NOT NULL,
                    experiment_name VARCHAR(100) NOT NULL,
                    variant VARCHAR(50) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_ab_test_logs_experiment "
                "ON ab_test_logs(experiment_name, variant)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_ab_test_logs_created "
                "ON ab_test_logs(created_at)"
            ))

            conn.commit()
            logger.info("Memory columns + analytics tables migration applied successfully.")
        except Exception as e:
            logger.warning(f"Migration warning (may already exist): {e}")


def init_admin_db(config: DictConfig) -> None:
    father_tg_id = os.getenv("FATHER_TG_ID")
    father_tg_tag = os.getenv("FATHER_TG_TAG")
    engine = create_engine(config["bot_db_connection"])

    Base.metadata.create_all(engine)
    migrate_memory_columns(engine)
    Session = sessionmaker(engine)

    try:
        with Session() as session:
            main_admin = Admin(
                tg_id=father_tg_id,
                tg_tag=father_tg_tag,
                can_add_info=True,
                can_add_new_admins=True,
                added_by_id=father_tg_id,
            )
            session.add(main_admin)
            session.commit()
    except IntegrityError:
        print("Admin database has been already created and initialized!")


def init_pgsql_docstore(store_config: DictConfig):
    docstore = instantiate(store_config, async_mode=False)
    docstore.create_schema()


# flake8: noqa: C901
def finditems(obj, key):
    found = []
    if isinstance(obj, (dict, DictConfig)):
        if key in obj:
            found.append(obj[key])
        for _, v in obj.items():
            if isinstance(v, (dict, DictConfig)):
                result = finditems(v, key)
                if result is not None:
                    found.extend(result)
            elif isinstance(v, (list, ListConfig)):
                for item in v:
                    result = finditems(item, key)
                    if result is not None:
                        found.extend(result)
    elif isinstance(obj, (list, ListConfig)):
        for item in obj:
            result = finditems(item, key)
            if result is not None:
                found.extend(result)
    return list(set(found))


@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(config: DictConfig) -> None:
    init_admin_db(config)


if __name__ == "__main__":
    main()
