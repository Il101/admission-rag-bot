import asyncio
import json
import logging
import time

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes
from telegram.error import BadRequest, RetryAfter

from bot.decorators import filter_banned, with_db_session
from bot.keyboards import suggested_questions_keyboard
from bot.utils import docs_to_sources_str, make_html_quote, remove_bot_command, sanitize_telegram_html
from bot.db import get_context_messages, add_chat_messages, get_user_memory
from bot.memory import (
    build_memory_context,
    get_current_date,
    update_journey_and_summary,
)
from crag.pipelines.base import documents_to_context_str

logger = logging.getLogger(__name__)

BUTTONS_MARKER = "---BUTTONS---"

# Streaming config
STREAM_EDIT_INTERVAL = 1.2  # seconds between edits
STREAM_MIN_CHARS = 80       # min new chars before edit


def parse_suggested_buttons(text: str) -> tuple:
    """Extract suggested questions from the LLM response.

    Returns (clean_text, list_of_questions).
    """
    if BUTTONS_MARKER not in text:
        return text, []

    parts = text.rsplit(BUTTONS_MARKER, 1)
    clean_text = parts[0].rstrip()
    buttons_raw = parts[1].strip()

    try:
        questions = json.loads(buttons_raw)
        if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
            return clean_text, questions
    except (json.JSONDecodeError, TypeError):
        pass

    return clean_text, []


async def infer_graph(
    graph: Runnable, question: str,
    only_docs: bool = False, user_data: dict = None,
    chat_history: str = "", memory_context: str = "",
) -> tuple:
    """Run RAG pipeline (retrieval only) and return graph state.
    
    With only_docs=True or do_generate=False, retrieves and grades documents
    without final generation. Returns the full graph state dict.
    """
    if user_data is None:
        user_data = {}

    response = await graph.ainvoke(
        {
            "question": question,
            "do_generate": not only_docs,
            "failed": False,
            "remaining_rewrites": 1,
            "user_data": user_data,
            "chat_history": chat_history,
            "memory_context": memory_context,
            "current_date": get_current_date(),
        }
    )
    return response


async def infer_retrieval_only(
    graph: Runnable, question: str, user_data: dict = None,
    chat_history: str = "", memory_context: str = "",
) -> dict:
    """Run retrieval + grading + possible rewriting, but NO generation.
    
    Returns graph state with documents and potentially rewritten question.
    """
    if user_data is None:
        user_data = {}

    response = await graph.ainvoke(
        {
            "question": question,
            "do_generate": False,
            "failed": False,
            "remaining_rewrites": 1,
            "user_data": user_data,
            "chat_history": chat_history,
            "memory_context": memory_context,
            "current_date": get_current_date(),
        }
    )
    return response


def _build_chat_history_str(history_list: list) -> str:
    if not history_list:
        return "История пуста."
    parts = []
    for msg in history_list:
        role = "🧑 Пользователь:" if msg["role"] == "user" else "🤖 Ассистент:"
        parts.append(f"{role} {msg['content']}")
    return "\n\n".join(parts)


async def _safe_edit_message(msg, text, parse_mode=None, reply_markup=None):
    """Edit a message, gracefully handling Telegram API errors."""
    try:
        await msg.edit_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
        return True
    except RetryAfter as e:
        await asyncio.sleep(e.retry_after)
        try:
            await msg.edit_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
            return True
        except Exception:
            return False
    except BadRequest as e:
        if "message is not modified" in str(e).lower():
            return True  # same content, no problem
        logger.warning(f"Edit failed: {e}")
        return False
    except Exception as e:
        logger.warning(f"Edit failed: {e}")
        return False


async def _stream_answer(
    msg, rag_chain, question: str, context_str: str,
    chat_history: str, memory_context: str, prefix: str = "",
):
    """Stream LLM generation, editing the Telegram message periodically.
    
    Returns the full accumulated response text.
    """
    accumulated = prefix
    last_edit_time = time.monotonic()
    last_edit_len = len(prefix)

    async for chunk in rag_chain.astream({
        "context": context_str,
        "question": question,
        "chat_history": chat_history,
        "memory_context": memory_context,
        "current_date": get_current_date(),
    }):
        accumulated += chunk
        now = time.monotonic()
        new_chars = len(accumulated) - last_edit_len

        # Edit periodically: every STREAM_EDIT_INTERVAL seconds AND at least STREAM_MIN_CHARS new
        if now - last_edit_time >= STREAM_EDIT_INTERVAL and new_chars >= STREAM_MIN_CHARS:
            # Strip BUTTONS marker from display during streaming
            display = accumulated.split(BUTTONS_MARKER)[0].rstrip()
            safe = sanitize_telegram_html(display + " ▌")
            if safe.strip():
                await _safe_edit_message(msg, safe, parse_mode=ParseMode.HTML)
            last_edit_len = len(accumulated)
            last_edit_time = now

    return accumulated


async def _send_response(context, chat_id, reply_to_id, text, suggested, user_data):
    """Send response with suggested question buttons (LLM-generated only)."""
    keyboard = None
    if suggested:
        keyboard = suggested_questions_keyboard(suggested)
        user_data["_suggested_questions"] = suggested

    # Sanitize text: strip all HTML tags Telegram doesn't support
    safe_text = sanitize_telegram_html(text)

    await context.bot.send_message(
        chat_id=chat_id,
        reply_to_message_id=reply_to_id,
        text=safe_text,
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


async def _load_memory(db_session, tg_id, onboarding_data):
    """Load all 3 memory layers and build context string."""
    memory = await get_user_memory(db_session, tg_id)
    history_list = await get_context_messages(db_session, tg_id)
    chat_history_str = _build_chat_history_str(history_list)
    memory_context_str = build_memory_context(
        memory["journey_state"], memory["conversation_summary"], onboarding_data
    )
    return memory, chat_history_str, memory_context_str


@with_db_session()
@filter_banned()
async def answer(
    update: Update, context: ContextTypes.DEFAULT_TYPE,
    graph: Runnable, rag_chain, llm: BaseLanguageModel, db_session, **kwargs
):
    question = remove_bot_command(
        update.effective_message.text, "ans", context.bot.name
    )

    if len(question) == 0:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.effective_message.id,
            text="Вы не задали вопрос.",
        )
        return

    chat_id = update.effective_chat.id
    reply_to_id = update.effective_message.id

    try:
        tg_id = update.effective_user.id
        onboarding_data = context.user_data.get("onboarding", {})
        memory, chat_history_str, memory_context_str = await _load_memory(
            db_session, tg_id, onboarding_data
        )

        # Phase 1: Send placeholder and run retrieval
        msg = await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_id,
            text="🔍 Ищу информацию...",
        )

        retrieval_result = await infer_retrieval_only(
            graph, question, user_data=onboarding_data,
            chat_history=chat_history_str, memory_context=memory_context_str,
        )

        docs = retrieval_result["documents"]
        actual_question = retrieval_result["question"]

        # Phase 2: Handle giveup or stream generation
        if retrieval_result.get("failed") or len(docs) == 0:
            # Giveup case — use giveup text
            giveup_text = retrieval_result.get(
                "generation",
                "К сожалению, я не смог найти релевантную информацию по этому запросу. 😔"
            )
            clean_text, suggested = parse_suggested_buttons(giveup_text)
            safe_text = sanitize_telegram_html(clean_text)

            keyboard = None
            if suggested:
                keyboard = suggested_questions_keyboard(suggested)
                context.user_data["_suggested_questions"] = suggested

            await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
            await add_chat_messages(db_session, tg_id, question, clean_text)
            return

        # Build prefix for rewritten question
        prefix = ""
        if question != actual_question:
            prefix = "Изменен вопрос/запрос на:\n" + make_html_quote(actual_question)

        # Phase 3: Stream generation
        context_str = documents_to_context_str(docs)
        full_response = await _stream_answer(
            msg, rag_chain, actual_question, context_str,
            chat_history_str, memory_context_str, prefix=prefix,
        )

        # Phase 4: Final edit with sources + buttons
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

        # Save to DB and update memory
        await add_chat_messages(db_session, tg_id, question, clean_text)
        await update_journey_and_summary(llm, db_session, tg_id, question, clean_text)

    except Exception:
        logger.exception("Error in answer handler")
        await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_id,
            text="⚠️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
        )


@with_db_session()
@filter_banned()
async def answer_to_replied(
    update: Update, context: ContextTypes.DEFAULT_TYPE,
    graph: Runnable, rag_chain, llm: BaseLanguageModel, db_session, **kwargs
):
    question = remove_bot_command(
        update.effective_message.reply_to_message.text, "ans_rep", context.bot.name
    )

    if len(question) == 0:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.effective_message.reply_to_message.id,
            text="Вы не задали вопрос.",
        )
        return

    chat_id = update.effective_chat.id
    reply_to_id = update.effective_message.reply_to_message.id

    try:
        tg_id = update.effective_user.id
        onboarding_data = context.user_data.get("onboarding", {})
        memory, chat_history_str, memory_context_str = await _load_memory(
            db_session, tg_id, onboarding_data
        )

        # Phase 1: Send placeholder and run retrieval
        msg = await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_id,
            text="🔍 Ищу информацию...",
        )

        retrieval_result = await infer_retrieval_only(
            graph, question, user_data=onboarding_data,
            chat_history=chat_history_str, memory_context=memory_context_str,
        )

        docs = retrieval_result["documents"]
        actual_question = retrieval_result["question"]

        # Phase 2: Handle giveup or stream generation
        if retrieval_result.get("failed") or len(docs) == 0:
            giveup_text = retrieval_result.get(
                "generation",
                "К сожалению, я не смог найти релевантную информацию по этому запросу. 😔"
            )
            clean_text, suggested = parse_suggested_buttons(giveup_text)
            safe_text = sanitize_telegram_html(clean_text)

            keyboard = None
            if suggested:
                keyboard = suggested_questions_keyboard(suggested)
                context.user_data["_suggested_questions"] = suggested

            await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
            await add_chat_messages(db_session, tg_id, question, clean_text)
            return

        # Build prefix for rewritten question
        prefix = ""
        if question != actual_question:
            prefix = "Изменен вопрос/запрос на:\n" + make_html_quote(actual_question)

        # Phase 3: Stream generation
        context_str = documents_to_context_str(docs)
        full_response = await _stream_answer(
            msg, rag_chain, actual_question, context_str,
            chat_history_str, memory_context_str, prefix=prefix,
        )

        # Phase 4: Final edit with sources + buttons
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
        await update_journey_and_summary(llm, db_session, tg_id, question, clean_text)

    except Exception:
        logger.exception("Error in answer_to_replied handler")
        await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_id,
            text="⚠️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
        )


@with_db_session()
@filter_banned()
async def retieve_docs(
    update: Update, context: ContextTypes.DEFAULT_TYPE, graph: Runnable, **kwargs
):
    question = remove_bot_command(
        update.effective_message.text, "docs", context.bot.name
    )

    if len(question) == 0:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.effective_message.id,
            text="Вы не задали вопрос.",
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        user_data = context.user_data.get("onboarding", {})
        response = await infer_graph(graph, question, only_docs=True, user_data=user_data)
        
        output = ""
        docs = response["documents"]
        if len(docs) > 0:
            sources_text = docs_to_sources_str(docs)
            output = "Источники/наиболее релевантные ссылки:\n" + sources_text
        else:
            output = "К сожалению, релевантных документов не найдено."

        await _send_response(
            context, update.effective_chat.id, update.effective_message.id,
            output, [], context.user_data,
        )
    except Exception:
        logger.exception("Error in retieve_docs handler")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.effective_message.id,
            text="⚠️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
        )


@with_db_session()
@filter_banned()
async def retieve_docs_to_replied(
    update: Update, context: ContextTypes.DEFAULT_TYPE, graph: Runnable, **kwargs
):
    question = remove_bot_command(
        update.effective_message.reply_to_message.text, "docs_rep", context.bot.name
    )

    if len(question) == 0:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.effective_message.reply_to_message.id,
            text="Вы не задали вопрос.",
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        user_data = context.user_data.get("onboarding", {})
        response = await infer_graph(graph, question, only_docs=True, user_data=user_data)
        
        output = ""
        docs = response["documents"]
        if len(docs) > 0:
            sources_text = docs_to_sources_str(docs)
            output = "Источники/наиболее релевантные ссылки:\n" + sources_text
        else:
            output = "К сожалению, релевантных документов не найдено."

        await _send_response(
            context, update.effective_chat.id, update.effective_message.reply_to_message.id,
            output, [], context.user_data,
        )
    except Exception:
        logger.exception("Error in retieve_docs_to_replied handler")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.effective_message.reply_to_message.id,
            text="⚠️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
        )
