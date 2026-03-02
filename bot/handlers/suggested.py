import logging
from functools import partial

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.decorators import filter_banned, with_db_session
from bot.handlers.rag import (
    _build_chat_history_str, _load_memory,
    _stream_answer, _safe_edit_message, parse_suggested_buttons,
)
from bot.keyboards import suggested_questions_keyboard
from bot.utils import docs_to_sources_str, make_html_quote, sanitize_telegram_html
from bot.db import add_chat_messages
from bot.memory import update_journey_and_summary
from crag.simple_rag import documents_to_context_str

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
            idx = int(query.data.split("_")[1])
        except (IndexError, ValueError):
            return

        suggested_questions = context.user_data.get("_suggested_questions", [])
        if idx >= len(suggested_questions):
            return

        question = suggested_questions[idx]

        try:
            tg_id = update.effective_user.id
            onboarding_data = context.user_data.get("onboarding", {})
            memory, chat_history_str, memory_context_str = await _load_memory(
                db_session, tg_id, onboarding_data
            )

            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"❓ {question}",
            )

            msg = await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="🔍 Ищу информацию...",
            )

            docs = await simple_rag.aretrieve(question)
            actual_question = question

            if len(docs) == 0:
                giveup_text = "К сожалению, я не смог найти релевантную информацию. 😔"
                clean_text, suggested = parse_suggested_buttons(giveup_text)
                safe_text = sanitize_telegram_html(clean_text)
                keyboard = None
                if suggested:
                    keyboard = suggested_questions_keyboard(suggested)
                    context.user_data["_suggested_questions"] = suggested
                await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
                await add_chat_messages(db_session, tg_id, question, clean_text)
                return

            prefix = ""
            context_str = documents_to_context_str(docs)
            
            full_response = await _stream_answer(
                msg, simple_rag, actual_question, context_str,
                chat_history_str, memory_context_str, prefix=prefix,
            )

            sources_text = docs_to_sources_str(docs)
            if sources_text:
                full_response += "\n\nИсточники/наиболее релевантные ссылки:\n" + sources_text

            clean_text, suggested = parse_suggested_buttons(full_response)
            safe_text = sanitize_telegram_html(clean_text)

            keyboard = None
            if suggested:
                keyboard = suggested_questions_keyboard(suggested)
                context.user_data["_suggested_questions"] = suggested

            await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

            await add_chat_messages(db_session, tg_id, question, clean_text)
            await update_journey_and_summary(simple_rag, db_session, tg_id, question, clean_text)

        except Exception:
            logger.exception("Error in suggested question handler")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
            )

    return partial(handle_suggested_question, simple_rag=simple_rag, db_session=db_session)
