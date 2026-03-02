import asyncio
import json
import logging
import time
import re

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
from crag.simple_rag import documents_to_context_str

logger = logging.getLogger(__name__)

# Streaming config
STREAM_EDIT_INTERVAL = 1.2  # seconds between edits
STREAM_MIN_CHARS = 80       # min new chars before edit


def parse_suggested_buttons(text: str) -> tuple:
    """Extract suggested questions from the structured JSON response.
    Returns (clean_text, list_of_questions).
    """
    logger.debug(f"Parsing suggested buttons from: {text[:200]}...")
    
    # Try to find a JSON block if it's wrapped in something
    json_text = text.strip()
    if "{" in json_text and "}" in json_text:
        try:
            start = json_text.find("{")
            end = json_text.rfind("}") + 1
            json_text = json_text[start:end]
        except:
            pass

    try:
        data = json.loads(json_text)
        if isinstance(data, dict):
            answer = data.get("answer", "")
            suggested = data.get("suggested_questions", [])
            if isinstance(suggested, list):
                return answer, suggested
        return text, []
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from response. Text starts with: {text[:100]}")
        # Fallback for non-JSON or partial JSON
        # Try to extract "answer" content
        ans_match = re.search(r'"answer":\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        # Try to extract "suggested_questions" array
        btns_match = re.search(r'"suggested_questions":\s*(\[[^\]]*\])', text, re.DOTALL)
        
        ans = text
        btns = []
        
        if ans_match:
            ans = ans_match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        
        if btns_match:
            try:
                btns = json.loads(btns_match.group(1))
            except:
                # Last resort regex for strings in the array
                btns = re.findall(r'"([^"]*)"', btns_match.group(1))
        
        return ans, btns


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
    msg, simple_rag, question: str, context_str: str,
    chat_history: str, memory_context: str, prefix: str = "",
):
    """Stream LLM generation (JSON format), editing the Telegram message periodically.
    Returns the full accumulated response JSON text.
    """
    accumulated = prefix
    last_edit_time = time.monotonic()
    last_edit_len = len(prefix)

    # Regex to extract partial answer from a JSON stream: {"answer": "..."}
    # We look for the "answer" key and capture its content until it closes or stream ends.
    answer_regex = re.compile(r'"answer":\s*"((?:[^"\\]|\\.)*)', re.DOTALL)

    async for chunk in simple_rag.astream_answer(
        question, context_str, chat_history, memory_context, get_current_date()
    ):
        accumulated += chunk
        now = time.monotonic()
        
        # Extract what we have of the answer so far
        match = answer_regex.search(accumulated)
        if match:
            display_text = match.group(1)
            # Safely unescape only JSON control characters, don't use unicode_escape on the whole string
            # as it breaks already decoded UTF-8 characters (Russian).
            display_text = display_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            
            new_chars = len(display_text) - last_edit_len

            # Edit periodically: every STREAM_EDIT_INTERVAL seconds AND at least STREAM_MIN_CHARS new
            if now - last_edit_time >= STREAM_EDIT_INTERVAL and new_chars >= STREAM_MIN_CHARS:
                safe = sanitize_telegram_html(display_text + " ▌")
                if safe.strip():
                    await _safe_edit_message(msg, safe, parse_mode=ParseMode.HTML)
                last_edit_len = len(display_text)
                last_edit_time = now

    return accumulated


async def _send_response(context, chat_id, reply_to_id, text, suggested, user_data):
    """Send response with suggested question buttons (LLM-generated only)."""
    keyboard = None
    if suggested:
        keyboard = suggested_questions_keyboard(suggested)
        user_data["_suggested_questions"] = suggested

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
    simple_rag, db_session, **kwargs
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

        msg = await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_id,
            text="🔍 Ищу информацию...",
        )

        docs = await simple_rag.aretrieve(question)
        actual_question = question

        if len(docs) == 0:
            giveup_text = "К сожалению, я не смог найти релевантную информацию по этому запросу. 😔"
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

        clean_text, suggested = parse_suggested_buttons(full_response)

        sources_text = docs_to_sources_str(docs)
        if sources_text:
            clean_text += "\n\nИсточники/наиболее релевантные ссылки:\n" + sources_text

        safe_text = sanitize_telegram_html(clean_text)

        keyboard = None
        if suggested:
            keyboard = suggested_questions_keyboard(suggested)
            context.user_data["_suggested_questions"] = suggested

        await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

        await add_chat_messages(db_session, tg_id, question, clean_text)
        await update_journey_and_summary(simple_rag, db_session, tg_id, question, clean_text)

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
    simple_rag, db_session, **kwargs
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

        msg = await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_id,
            text="🔍 Ищу информацию...",
        )

        docs = await simple_rag.aretrieve(question)
        actual_question = question

        if len(docs) == 0:
            giveup_text = "К сожалению, я не смог найти релевантную информацию по этому запросу. 😔"
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

        clean_text, suggested = parse_suggested_buttons(full_response)

        sources_text = docs_to_sources_str(docs)
        if sources_text:
            clean_text += "\n\nИсточники/наиболее релевантные ссылки:\n" + sources_text

        safe_text = sanitize_telegram_html(clean_text)

        keyboard = None
        if suggested:
            keyboard = suggested_questions_keyboard(suggested)
            context.user_data["_suggested_questions"] = suggested

        await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

        await add_chat_messages(db_session, tg_id, question, clean_text)
        await update_journey_and_summary(simple_rag, db_session, tg_id, question, clean_text)

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
    update: Update, context: ContextTypes.DEFAULT_TYPE, simple_rag, **kwargs
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
        docs = await simple_rag.aretrieve(question)
        
        output = ""
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
    update: Update, context: ContextTypes.DEFAULT_TYPE, simple_rag, **kwargs
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
        docs = await simple_rag.aretrieve(question)
        
        output = ""
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
