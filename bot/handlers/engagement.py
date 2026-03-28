"""
Proactive re-engagement handler.

Sends a single follow-up message to users who have been idle for 48+ hours.
If they don't respond, we don't bother them again (no spamming).
"""

import logging
from telegram.ext import Application

from bot.db import (
    get_idle_users_for_reengagement,
    get_user_memory,
    mark_reengagement_sent,
)
from bot.keyboards import suggested_questions_keyboard
from bot.memory import JOURNEY_STAGES, get_fallback_buttons

logger = logging.getLogger(__name__)


async def send_reengagement_messages(
    application: Application,
    db_session_factory,
    idle_hours: int = 48,
    max_users: int = 20,
) -> int:
    """Send re-engagement messages to idle users.

    Args:
        application: Telegram bot application
        db_session_factory: Async session factory
        idle_hours: Hours of inactivity before triggering re-engagement
        max_users: Maximum users to process in one batch (to avoid rate limits)

    Returns:
        Number of messages successfully sent
    """
    sent_count = 0

    async with db_session_factory() as session:
        idle_users = await get_idle_users_for_reengagement(
            session, idle_hours=idle_hours, limit=max_users
        )

        if not idle_users:
            logger.debug("No idle users found for re-engagement")
            return 0

        logger.info(f"Found {len(idle_users)} idle users for re-engagement")

        for user in idle_users:
            try:
                # Build personalized message based on journey state
                message, buttons = _build_reengagement_message(user)

                # Send the message
                keyboard = suggested_questions_keyboard(buttons) if buttons else None
                await application.bot.send_message(
                    chat_id=user.tg_id,
                    text=message,
                    parse_mode="HTML",
                    reply_markup=keyboard,
                )

                # Mark as sent (we only send ONCE per idle period)
                await mark_reengagement_sent(session, user.tg_id)
                sent_count += 1
                logger.info(f"Re-engagement sent to user {user.tg_id}")

            except Exception as e:
                # User might have blocked the bot, deleted account, etc.
                # Mark as sent anyway to avoid retrying forever
                try:
                    await mark_reengagement_sent(session, user.tg_id)
                except Exception:
                    pass
                logger.warning(f"Failed to send re-engagement to {user.tg_id}: {e}")

    return sent_count


def _build_reengagement_message(user) -> tuple[str, list[str]]:
    """Build a personalized re-engagement message based on user's journey state.

    Returns:
        Tuple of (message_text, suggested_buttons)
    """
    journey_state = user.journey_state or {}
    summary = user.conversation_summary or ""

    # Find the most relevant pending stage
    pending_stages = []
    discussed_stages = []
    for stage_id, status in journey_state.items():
        if stage_id.startswith("_"):
            continue
        stage_info = JOURNEY_STAGES.get(stage_id)
        if not stage_info:
            continue
        if status == "discussed":
            discussed_stages.append(stage_info)
        else:
            pending_stages.append((stage_id, stage_info))

    # Build message based on progress
    if not discussed_stages:
        # User started but didn't discuss anything meaningful
        message = (
            "👋 Привет! Мы начали разговор, но не успели обсудить детали.\n\n"
            "Готов помочь с вопросами о поступлении в Австрию!"
        )
        buttons = [
            "📋 С чего мне начать?",
            "📄 Какие документы нужны?",
            "🏫 Какие вузы мне подходят?",
        ]
    elif len(discussed_stages) <= 3:
        # Early in the journey
        next_stage = pending_stages[0][1] if pending_stages else None
        if next_stage:
            message = (
                f"👋 Привет! Мы остановились на теме «{discussed_stages[-1]['label']}».\n\n"
                f"Следующий логичный шаг — <b>{next_stage['label']}</b>.\n"
                "Готов продолжить?"
            )
            buttons = [next_stage["question"]] + get_fallback_buttons(journey_state)[:2]
        else:
            message = (
                f"👋 Привет! Мы обсудили {len(discussed_stages)} тем.\n\n"
                "Есть вопросы или нужна помощь с чем-то ещё?"
            )
            buttons = get_fallback_buttons(journey_state)
    else:
        # Advanced user
        remaining = len(pending_stages)
        if remaining > 0:
            message = (
                f"👋 Ты уже прошёл большую часть пути — обсудили {len(discussed_stages)} из {len(JOURNEY_STAGES)} этапов!\n\n"
                f"Осталось: {', '.join([s[1]['label'] for s in pending_stages[:3]])}.\n"
                "Продолжим?"
            )
            buttons = [s[1]["question"] for s in pending_stages[:3]]
        else:
            message = (
                "👋 Привет! Мы обсудили все основные этапы поступления.\n\n"
                "Если есть дополнительные вопросы — всегда рад помочь!"
            )
            buttons = [
                "📋 Обнови мой чек-лист",
                "❓ У меня ещё вопрос",
            ]

    return message, buttons


async def setup_reengagement_job(application: Application, db_session_factory):
    """Set up a periodic job to send re-engagement messages.

    Runs every 6 hours to catch users who have been idle for 48+ hours.
    """
    job_queue = application.job_queue
    if not job_queue:
        logger.warning("Job queue not available, re-engagement disabled")
        return

    async def reengagement_callback(context):
        try:
            sent = await send_reengagement_messages(
                application,
                db_session_factory,
                idle_hours=48,
                max_users=20,
            )
            if sent > 0:
                logger.info(f"Re-engagement batch complete: {sent} messages sent")
        except Exception as e:
            logger.exception(f"Re-engagement job failed: {e}")

    # Run every 6 hours, starting 1 hour after bot start
    job_queue.run_repeating(
        reengagement_callback,
        interval=6 * 60 * 60,  # 6 hours
        first=60 * 60,  # First run after 1 hour
        name="reengagement_job",
    )
    logger.info("Re-engagement job scheduled (every 6 hours)")
