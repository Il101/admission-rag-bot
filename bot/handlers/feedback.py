"""Handler for user feedback (👍/👎) on bot responses."""

import logging
from functools import partial

from telegram import Update
from telegram.ext import ContextTypes

from bot.decorators import filter_banned, with_db_session
from bot.db import add_feedback

logger = logging.getLogger(__name__)


def build_feedback_handler(db_session):
    """Build a callback handler for feedback buttons."""

    @with_db_session()
    @filter_banned()
    async def handle_feedback(
        update: Update, context: ContextTypes.DEFAULT_TYPE,
        db_session, **kwargs
    ):
        query = update.callback_query

        # Parse callback data: fb_{msg_id}_{rating}
        try:
            parts = query.data.split("_")
            msg_id = int(parts[1])
            rating = int(parts[2])
        except (IndexError, ValueError):
            await query.answer("⚠️ Ошибка")
            return

        if rating not in (1, -1):
            await query.answer("⚠️ Ошибка")
            return

        tg_id = update.effective_user.id

        # Retrieve question/answer from in-memory store
        qa_map = context.user_data.get("_qa_map", {})
        qa = qa_map.get(msg_id, {})
        question = qa.get("question", "")
        answer_text = qa.get("answer", "")

        try:
            await add_feedback(
                db_session, tg_id, msg_id, question, answer_text, rating,
            )

            if rating == 1:
                await query.answer("Спасибо за отзыв! 👍")
            else:
                await query.answer("Спасибо, учтём! Мы работаем над улучшением 🙏")

            # Update keyboard: remove feedback row, keep suggested questions
            if query.message and query.message.reply_markup:
                old_rows = query.message.reply_markup.inline_keyboard
                # Keep only suggestion rows (those starting with 💬)
                new_rows = [
                    row for row in old_rows
                    if row and row[0].callback_data
                    and not row[0].callback_data.startswith("fb_")
                ]
                # Add a "rated" indicator
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                label = "✅ Спасибо за оценку!" if rating == 1 else "📝 Учтём, спасибо!"
                new_rows.append([InlineKeyboardButton(label, callback_data="noop")])
                try:
                    await query.message.edit_reply_markup(
                        reply_markup=InlineKeyboardMarkup(new_rows)
                    )
                except Exception:
                    pass  # Telegram may reject if nothing changed

            logger.info(
                "[User %s] Feedback: %s for msg %s",
                tg_id, "👍" if rating == 1 else "👎", msg_id,
            )

        except Exception:
            logger.exception("Failed to save feedback")
            await query.answer("⚠️ Ошибка при сохранении отзыва")

    return partial(handle_feedback, db_session=db_session)
