import logging
from functools import partial

from telegram import Update
from telegram.ext import ContextTypes

from bot.decorators import filter_banned, with_db_session
from bot.db import get_user_memory
from bot.handlers.rag import _handle_question_core, parse_suggested_buttons

logger = logging.getLogger(__name__)


def build_suggested_handler(simple_rag, db_session):
    """Build a callback handler for suggested question buttons."""

    @with_db_session()
    @filter_banned()
    async def handle_suggested_question(
        update: Update, context: ContextTypes.DEFAULT_TYPE,
        simple_rag, db_session, **kwargs
    ):
        query = update.callback_query
        await query.answer()

        try:
            data_parts = query.data.split("_")
            if data_parts[0] == "sq":
                # New format: sq_{msg_id}_{idx}
                msg_id = int(data_parts[1])
                idx = int(data_parts[2])
            else:
                # Legacy format: suggest_{idx}
                msg_id = None
                idx = int(data_parts[1])
        except (IndexError, ValueError):
            return

        # Try message-specific lookup first
        suggested_questions = []
        if msg_id:
            msg_suggestions = context.user_data.get("_message_suggestions", {})
            suggested_questions = msg_suggestions.get(msg_id, [])

        # Fall back to global in-memory (latest response)
        if not suggested_questions:
            suggested_questions = context.user_data.get("_suggested_questions", [])

        # Fall back to DB if user_data was lost (e.g. after bot restart)
        if not suggested_questions:
            try:
                tg_id_for_lookup = update.effective_user.id
                memory = await get_user_memory(db_session, tg_id_for_lookup)
                suggested_questions = (
                    (memory.get("journey_state") or {}).get("_last_suggested", [])
                )
            except Exception:
                logger.warning("Could not load suggested questions from DB", exc_info=True)

        if idx >= len(suggested_questions):
            return

        question = suggested_questions[idx]

        try:
            tg_id = update.effective_user.id
            onboarding_data = context.user_data.get("onboarding", {})

            # Echo the button text so user sees what was selected
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"❓ {question}",
            )

            await _handle_question_core(
                context=context,
                simple_rag=simple_rag,
                db_session=db_session,
                chat_id=update.effective_chat.id,
                reply_to_id=None,
                tg_id=tg_id,
                question=question,
                onboarding_data=onboarding_data,
                db_session_factory=kwargs.get("db_session_factory"),
            )

        except Exception:
            logger.exception("Error in suggested question handler")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
            )

    return partial(handle_suggested_question, simple_rag=simple_rag, db_session=db_session)
