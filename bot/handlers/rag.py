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
from bot.keyboards import suggested_questions_keyboard, combined_keyboard
from bot.utils import docs_to_sources_str, make_html_quote, remove_bot_command, sanitize_telegram_html
from bot.db import get_context_messages, add_chat_messages, get_user_memory, get_user, add_pipeline_log
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

# Country → country_scope code mapping (module-level constant for easy extension)
COUNTRY_TO_SCOPE: dict[str, str] = {
    "россия": "RU", "russia": "RU", "ru": "RU",
    "украина": "UA", "ukraine": "UA", "ua": "UA",
    "беларусь": "BY", "belarus": "BY", "by": "BY",
    "казахстан": "KZ", "kazakhstan": "KZ", "kz": "KZ",
}

# Level → scope code mapping
LEVEL_TO_SCOPE: dict[str, str] = {
    "bachelor": "bachelor",
    "master": "master",
    "phd": "phd",
}


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
        # Fallback: attempt regex extraction
        ans_match = re.search(r'"answer":\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        btns_match = re.search(r'"suggested_questions":\s*(\[[^\]]*\])', text, re.DOTALL)
        
        ans = text
        btns = []
        
        if ans_match:
            ans = ans_match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        
        if btns_match:
            try:
                btns = json.loads(btns_match.group(1))
            except:
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
    answer_regex = re.compile(r'"answer":\s*"((?:[^"\\]|\\.)*)', re.DOTALL)

    async for chunk in simple_rag.astream_answer(
        question, context_str, chat_history, memory_context, get_current_date()
    ):
        accumulated += chunk
        now = time.monotonic()
        
        match = answer_regex.search(accumulated)
        if match:
            display_text = match.group(1)
            display_text = display_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            
            new_chars = len(display_text) - last_edit_len

            if now - last_edit_time >= STREAM_EDIT_INTERVAL and new_chars >= STREAM_MIN_CHARS:
                safe = sanitize_telegram_html(display_text + " ▌")
                if safe.strip():
                    await _safe_edit_message(msg, safe, parse_mode=ParseMode.HTML)
                last_edit_len = len(display_text)
                last_edit_time = now

    return accumulated


async def _ensure_onboarding_loaded(context, db_session, tg_id: int) -> dict:
    """Lazy-load user profile from DB into context.user_data['onboarding'].
    
    context.user_data is in-memory and is lost on bot restart.
    This restores it transparently from the DB on the first request after restart.
    """
    onboarding = context.user_data.get("onboarding", {})
    # If already populated (typical case during same session), return immediately
    if onboarding.get("targetLevel") or onboarding.get("germanLevel"):
        return onboarding

    # Load from DB and map snake_case column names → camelCase keys used in code
    user = await get_user(db_session, tg_id)
    if user:
        onboarding = {
            "countryScope": user.country or "RU",
            "document": user.document_type,
            "targetLevel": user.target_level,
            "germanLevel": user.german_level,
            "englishLevel": user.english_level,
        }
        # Remove None values so .get() checks still work cleanly
        onboarding = {k: v for k, v in onboarding.items() if v is not None}
        context.user_data["onboarding"] = onboarding
        logger.info(f"[User {tg_id}] Restored onboarding profile from DB: {onboarding}")
    return onboarding


async def _load_memory(db_session, tg_id, onboarding_data):
    """Load all 3 memory layers and build context string."""
    memory = await get_user_memory(db_session, tg_id)
    history_list = await get_context_messages(db_session, tg_id)
    chat_history_str = _build_chat_history_str(history_list)
    memory_context_str = build_memory_context(
        memory["journey_state"], memory["conversation_summary"], onboarding_data
    )
    return memory, chat_history_str, memory_context_str


def _build_user_filters(onboarding_data: dict) -> dict:
    """Build metadata filters from user's onboarding profile.
    
    Filters by country_scope and soft-boosts by level.
    """
    filters = {}
    
    country = onboarding_data.get("country")
    if country:
        code = COUNTRY_TO_SCOPE.get(country.lower().strip(), "")
        if code:
            filters["country_scope"] = code
    
    target_level = onboarding_data.get("targetLevel")
    if target_level:
        code = LEVEL_TO_SCOPE.get(target_level.lower().strip(), "")
        if code:
            filters["level"] = code

    return filters


def _save_suggested(context, db_session, tg_id, suggested: list, msg_id: int = None):
    """Save suggested questions both in-memory and schedule DB persist."""
    # Store globally for legacy buttons and as "latest"
    context.user_data["_suggested_questions"] = suggested
    
    # Store specifically for this message if ID is provided
    if msg_id:
        # Use a dict to store multiple recent suggestion sets
        if "_message_suggestions" not in context.user_data:
            context.user_data["_message_suggestions"] = {}
        
        context.user_data["_message_suggestions"][msg_id] = suggested
        
        # Simple cleanup: keep only last 10 messages' suggestions to save memory
        if len(context.user_data["_message_suggestions"]) > 10:
            oldest_keys = sorted(context.user_data["_message_suggestions"].keys())[:-10]
            for k in oldest_keys:
                context.user_data["_message_suggestions"].pop(k, None)

    # Persist asynchronously so they survive bot restarts (only globally for now)
    if suggested and db_session and tg_id:
        asyncio.create_task(_persist_suggested(db_session, tg_id, suggested))


async def _persist_suggested(db_session, tg_id: int, suggested: list):
    """Persist suggested questions into journey_state._last_suggested."""
    from bot.db import get_user_memory, update_user_memory
    try:
        memory = await get_user_memory(db_session, tg_id)
        state = memory.get("journey_state") or {}
        state["_last_suggested"] = suggested
        await update_user_memory(db_session, tg_id, journey_state=state)
    except Exception:
        logger.warning("Failed to persist suggested questions to DB", exc_info=True)


def _store_qa(context, msg_id: int, question: str, answer_text: str):
    """Store question/answer pair for feedback tracking."""
    if "_qa_map" not in context.user_data:
        context.user_data["_qa_map"] = {}
    context.user_data["_qa_map"][msg_id] = {
        "question": question[:500],
        "answer": answer_text[:500],
    }
    qa = context.user_data["_qa_map"]
    if len(qa) > 20:
        for k in sorted(qa.keys())[:-20]:
            qa.pop(k, None)


def _fmt_timings(t: dict) -> str:
    """Format pipeline step timings for logging."""
    return " | ".join(f"{k}: {v:.2f}s" for k, v in t.items())


async def _handle_question_core(
    context,
    simple_rag,
    db_session,
    chat_id: int,
    reply_to_id: int,
    tg_id: int,
    question: str,
    onboarding_data: dict,
):
    """Core logic: memory → rewrite → cache check → retrieve → grade → stream → send.
    
    Shared by answer(), answer_to_replied(), and handle_suggested_question().
    """
    timings = {}
    pipeline_start = time.monotonic()

    # Always restore onboarding from DB if lost after bot restart
    onboarding_data = await _ensure_onboarding_loaded(context, db_session, tg_id)

    memory, chat_history_str, memory_context_str = await _load_memory(
        db_session, tg_id, onboarding_data
    )

    msg = await context.bot.send_message(
        chat_id=chat_id,
        reply_to_message_id=reply_to_id,
        text="🔍 Ищу информацию...",
    )

    # 1. Rewrite query for better vector search (resolves anaphora, adds keywords)
    t0 = time.monotonic()
    actual_question = await simple_rag.arewrite_query(
        question, chat_history_str, memory_context_str
    )
    timings["rewrite"] = time.monotonic() - t0

    # 1a. Check semantic answer cache
    t0 = time.monotonic()
    cached_response = await simple_rag.get_cached_answer(actual_question)
    timings["cache_check"] = time.monotonic() - t0

    if cached_response:
        clean_text, suggested = parse_suggested_buttons(cached_response)
        safe_text = sanitize_telegram_html(clean_text)
        keyboard = combined_keyboard(suggested, msg.message_id) if suggested else None
        await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        if suggested:
            _save_suggested(context, db_session, tg_id, suggested, msg_id=msg.message_id)
        _store_qa(context, msg.message_id, question, clean_text)
        await add_chat_messages(db_session, tg_id, question, clean_text)
        asyncio.create_task(
            update_journey_and_summary(simple_rag, db_session, tg_id, question, clean_text)
        )
        timings["total"] = time.monotonic() - pipeline_start
        logger.info("[User %s] Cache hit | %s", tg_id, _fmt_timings(timings))
        asyncio.create_task(add_pipeline_log(
            db_session, tg_id, question, actual_question,
            cache_hit=True, docs_retrieved=0, docs_after_grading=0, timings=timings,
        ))
        return

    # 2. Retrieve candidates
    t0 = time.monotonic()
    user_filters = _build_user_filters(onboarding_data)
    docs = await simple_rag.aretrieve(actual_question, user_filters=user_filters)
    timings["retrieve"] = time.monotonic() - t0

    # 3. Grade for relevance (CRAG filtering)
    t0 = time.monotonic()
    docs = await simple_rag.agrade_documents(actual_question, docs)
    timings["grade"] = time.monotonic() - t0

    if len(docs) == 0:
        giveup_text = "К сожалению, я не смог найти релевантную информацию по этому запросу. 😔"
        clean_text, suggested = parse_suggested_buttons(giveup_text)
        safe_text = sanitize_telegram_html(clean_text)

        keyboard = combined_keyboard(suggested, msg.message_id) if suggested else None
        if suggested:
            _save_suggested(context, db_session, tg_id, suggested, msg_id=msg.message_id)

        await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        _store_qa(context, msg.message_id, question, clean_text)
        await add_chat_messages(db_session, tg_id, question, clean_text)
        timings["total"] = time.monotonic() - pipeline_start
        logger.info("[User %s] No relevant docs | %s", tg_id, _fmt_timings(timings))
        asyncio.create_task(add_pipeline_log(
            db_session, tg_id, question, actual_question,
            cache_hit=False, docs_retrieved=0, docs_after_grading=0, timings=timings,
        ))
        return

    context_str = documents_to_context_str(docs)
    
    # 4. Stream answer
    t0 = time.monotonic()
    full_response = await _stream_answer(
        msg, simple_rag, actual_question, context_str,
        chat_history_str, memory_context_str,
    )
    timings["generate"] = time.monotonic() - t0

    # Cache answer for future similar questions
    asyncio.create_task(simple_rag.cache_answer(actual_question, full_response))

    clean_text, suggested = parse_suggested_buttons(full_response)

    sources_text = docs_to_sources_str(docs)
    if sources_text:
        clean_text += "\n\nИсточники/наиболее релевантные ссылки:\n" + sources_text

    safe_text = sanitize_telegram_html(clean_text)

    # Combined keyboard: suggested questions + feedback (👍/👎)
    keyboard = combined_keyboard(suggested, msg.message_id) if suggested else None
    await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    if suggested:
        _save_suggested(context, db_session, tg_id, suggested, msg_id=msg.message_id)

    # Store Q&A for feedback tracking
    _store_qa(context, msg.message_id, question, clean_text)

    # 5. Persist conversation + update memory
    await add_chat_messages(db_session, tg_id, question, clean_text)
    asyncio.create_task(
        update_journey_and_summary(simple_rag, db_session, tg_id, question, clean_text)
    )

    timings["total"] = time.monotonic() - pipeline_start
    logger.info(
        "[User %s] Pipeline complete (%d docs) | %s",
        tg_id, len(docs), _fmt_timings(timings),
    )
    asyncio.create_task(add_pipeline_log(
        db_session, tg_id, question, actual_question,
        cache_hit=False, docs_retrieved=len(docs), docs_after_grading=len(docs),
        timings=timings,
    ))


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

    try:
        await _handle_question_core(
            context=context,
            simple_rag=simple_rag,
            db_session=db_session,
            chat_id=update.effective_chat.id,
            reply_to_id=update.effective_message.id,
            tg_id=update.effective_user.id,
            question=question,
            onboarding_data=context.user_data.get("onboarding", {}),
        )
    except Exception:
        logger.exception("Error in answer handler")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.effective_message.id,
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

    try:
        await _handle_question_core(
            context=context,
            simple_rag=simple_rag,
            db_session=db_session,
            chat_id=update.effective_chat.id,
            reply_to_id=update.effective_message.reply_to_message.id,
            tg_id=update.effective_user.id,
            question=question,
            onboarding_data=context.user_data.get("onboarding", {}),
        )
    except Exception:
        logger.exception("Error in answer_to_replied handler")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.effective_message.reply_to_message.id,
            text="⚠️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
        )


@with_db_session()
@filter_banned()
async def retrieve_docs(
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
        logger.exception("Error in retrieve_docs handler")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.effective_message.id,
            text="⚠️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
        )


@with_db_session()
@filter_banned()
async def retrieve_docs_to_replied(
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
        logger.exception("Error in retrieve_docs_to_replied handler")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.effective_message.reply_to_message.id,
            text="⚠️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
        )


# ── Backward-compat aliases for old typo names (used in app.py) ──────────────
retieve_docs = retrieve_docs
retieve_docs_to_replied = retrieve_docs_to_replied
