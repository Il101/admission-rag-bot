"""
Analytics script — export pipeline logs, feedback, and user data to CSV.

Provides:
  - Pipeline performance report (latency breakdown, cache hit rate)
  - Feedback report (positive/negative answers for manual review)
  - User activity report
  - Negative feedback export for quality review

Usage:
    python analytics.py [--days 30] [--output-dir outputs/analytics]
"""

import asyncio
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bot.env  # noqa: F401
import yaml
from sqlalchemy import create_engine, text, select, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from bot.db import (
    Base, User, Message, Feedback, PipelineLog,
    get_analytics_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_db_url() -> str:
    """Read DB URL from config and resolve env vars."""
    config_path = Path(__file__).resolve().parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    db_url = config.get("bot_db_connection", "")
    db_url = (
        db_url.replace("${oc.env:POSTGRES_USER}", os.environ.get("POSTGRES_USER", ""))
        .replace("${oc.env:POSTGRES_PASSWORD}", os.environ.get("POSTGRES_PASSWORD", ""))
        .replace("${oc.env:POSTGRES_HOST}", os.environ.get("POSTGRES_HOST", ""))
        .replace("${oc.env:POSTGRES_DB}", os.environ.get("POSTGRES_DB", ""))
    )
    return db_url


async def export_pipeline_logs(session: AsyncSession, days: int, output_dir: str):
    """Export pipeline logs to CSV."""
    cutoff = datetime.now() - timedelta(days=days)
    result = await session.execute(
        select(PipelineLog)
        .where(PipelineLog.created_at >= cutoff)
        .order_by(PipelineLog.created_at.desc())
    )
    logs = result.scalars().all()

    path = os.path.join(output_dir, "pipeline_logs.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "tg_id", "question", "rewritten_question",
            "cache_hit", "docs_retrieved", "docs_after_grading",
            "t_rewrite", "t_retrieve", "t_grade", "t_generate", "t_total",
        ])
        for log in logs:
            writer.writerow([
                log.created_at.isoformat() if log.created_at else "",
                log.tg_id, log.question, log.rewritten_question,
                log.cache_hit, log.docs_retrieved, log.docs_after_grading,
                round(log.t_rewrite or 0, 3), round(log.t_retrieve or 0, 3),
                round(log.t_grade or 0, 3), round(log.t_generate or 0, 3),
                round(log.t_total or 0, 3),
            ])

    logger.info("Pipeline logs: %d rows → %s", len(logs), path)
    return logs


async def export_feedback(session: AsyncSession, days: int, output_dir: str):
    """Export feedback to CSV."""
    cutoff = datetime.now() - timedelta(days=days)
    result = await session.execute(
        select(Feedback)
        .where(Feedback.created_at >= cutoff)
        .order_by(Feedback.created_at.desc())
    )
    feedbacks = result.scalars().all()

    path = os.path.join(output_dir, "feedback.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "tg_id", "message_id", "rating", "question", "answer"])
        for fb in feedbacks:
            writer.writerow([
                fb.created_at.isoformat() if fb.created_at else "",
                fb.tg_id, fb.message_id,
                "👍" if fb.rating == 1 else "👎",
                fb.question, fb.answer,
            ])

    logger.info("Feedback: %d rows → %s", len(feedbacks), path)

    # Separate file for negative feedback (easy review)
    negatives = [fb for fb in feedbacks if fb.rating == -1]
    if negatives:
        neg_path = os.path.join(output_dir, "negative_feedback.csv")
        with open(neg_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "tg_id", "question", "answer"])
            for fb in negatives:
                writer.writerow([
                    fb.created_at.isoformat() if fb.created_at else "",
                    fb.tg_id, fb.question, fb.answer,
                ])
        logger.info("Negative feedback: %d rows → %s", len(negatives), neg_path)

    return feedbacks


async def export_users(session: AsyncSession, days: int, output_dir: str):
    """Export user activity to CSV."""
    cutoff = datetime.now() - timedelta(days=days)
    result = await session.execute(
        select(User).order_by(User.last_active.desc())
    )
    users = result.scalars().all()

    path = os.path.join(output_dir, "users.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tg_id", "country", "target_level", "german_level", "english_level",
            "stages_discussed", "created_at", "last_active",
        ])
        for u in users:
            stages_discussed = 0
            if u.journey_state:
                stages_discussed = sum(
                    1 for k, v in u.journey_state.items()
                    if not k.startswith("_") and v == "discussed"
                )
            writer.writerow([
                u.tg_id, u.country, u.target_level, u.german_level, u.english_level,
                stages_discussed,
                u.created_at.isoformat() if u.created_at else "",
                u.last_active.isoformat() if u.last_active else "",
            ])

    logger.info("Users: %d rows → %s", len(users), path)
    return users


async def print_summary(session: AsyncSession, days: int):
    """Print a quick analytics summary to console."""
    try:
        data = await get_analytics_summary(session, days=days)
    except Exception as e:
        logger.warning("Could not generate full analytics (table may not exist yet): %s", e)
        return

    u = data["users"]
    q = data["queries"]
    fb = data["feedback"]

    satisfaction = "-"
    if fb["total"] > 0:
        satisfaction = f"{fb['positive'] / fb['total'] * 100:.0f}%"

    print("\n" + "=" * 50)
    print(f"📊 ANALYTICS SUMMARY (last {days} days)")
    print("=" * 50)
    print(f"\n👥 Users: {u['total']} total, {u['active_last_period']} active")
    print(f"\n⚡ Pipeline:")
    print(f"   Queries: {q['total']}")
    print(f"   Cache hits: {q['cache_hits']} ({q['cache_hit_rate']:.1%})")
    print(f"   Avg latency: {q['avg_latency_s']}s")
    print(f"   P95 latency: {q['p95_latency_s']}s")
    print(f"   Avg docs/query: {q['avg_docs_per_query']}")
    print(f"\n👍 Feedback:")
    print(f"   👍 {fb['positive']}  👎 {fb['negative']}  (total: {fb['total']})")
    print(f"   Satisfaction: {satisfaction}")
    print(f"\n💬 Messages: {data['messages']['total_last_period']}")
    print("=" * 50)


async def run(days: int, output_dir: str):
    db_url = get_db_url()
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    os.makedirs(output_dir, exist_ok=True)

    async with async_session() as session:
        await print_summary(session, days)
        await export_pipeline_logs(session, days, output_dir)
        await export_feedback(session, days, output_dir)
        await export_users(session, days, output_dir)

    await engine.dispose()
    logger.info("\n✅ All exports saved to: %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Export bot analytics to CSV")
    parser.add_argument("--days", type=int, default=30, help="Export data for the last N days")
    parser.add_argument("--output-dir", default="outputs/analytics", help="Output directory")
    args = parser.parse_args()

    asyncio.run(run(args.days, args.output_dir))


if __name__ == "__main__":
    main()
