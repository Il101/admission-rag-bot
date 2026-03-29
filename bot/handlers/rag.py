import asyncio
import json
import logging
import os
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
from crag.observability import (
    is_observability_enabled,
    log_pipeline_metrics,
    log_routing_decision,
    increment_routing_stat,
    trace_full_pipeline,
)
from crag.simple_rag import documents_to_context_str
from crag.pipeline import PipelineContext, create_default_pipeline
from crag.router import (
    classify_intent,
    Intent,
    should_use_rag,
    should_use_tools,
    is_chitchat,
    get_chitchat_response,
)
from crag.tools import execute_tool, get_tool_schemas

logger = logging.getLogger(__name__)

# Streaming config
STREAM_EDIT_INTERVAL = 1.2  # seconds between edits
STREAM_MIN_CHARS = 80       # min new chars before edit

# Processing phases for user feedback
PROCESSING_PHASES = {
    "search": "🔍 Ищу информацию...",
    "analyze": "🧠 Анализирую контекст...",
    "generate": "✍️ Формирую ответ...",
}


async def _update_processing_phase(msg, phase: str) -> bool:
    """Update the processing status message with current phase.

    Returns True if update succeeded, False otherwise.
    """
    phase_text = PROCESSING_PHASES.get(phase, PROCESSING_PHASES["search"])
    try:
        await msg.edit_text(phase_text)
        return True
    except Exception:
        return False


def _use_new_pipeline_enabled() -> bool:
    """Feature flag for gradual migration to crag.pipeline orchestrator."""
    return os.getenv("USE_NEW_PIPELINE", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _use_pipeline_shadow_enabled() -> bool:
    """Enable background shadow run of pipeline v2 while serving legacy path."""
    return os.getenv("USE_NEW_PIPELINE_SHADOW", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _use_routing_enabled() -> bool:
    """Enable smart routing with intent classification (bypasses RAG when not needed)."""
    return os.getenv("USE_ROUTING", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _safe_stream_plain_enabled() -> bool:
    """Use plain text for partial streaming edits to avoid Telegram HTML parse errors."""
    return os.getenv("SAFE_STREAM_PLAIN", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _get_rerank_soft_timeout_sec() -> float:
    """Soft timeout for re-ranking stage; 0 disables timeout."""
    raw = os.getenv("RERANK_SOFT_TIMEOUT_SEC", "0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, value)


def _strip_html_to_plain(text: str) -> str:
    """Convert HTML-like text to plain text for Telegram fallback rendering."""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(?:p|div|li|ul|ol|details|summary|section|article|header|footer|h[1-6]|tr|td|th|table|thead|tbody)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_spacing_entities(text: str) -> str:
    """Decode HTML spacing entities and normalize narrow/non-breaking spaces."""
    import html

    normalized = html.unescape(text or "")
    normalized = normalized.replace("\u00A0", " ").replace("\u202F", " ")
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized


def _format_deadline_tables_as_cards(text: str) -> str:
    """Render deadline table-like blocks as compact cards for Telegram readability."""
    if not text:
        return text

    t = _normalize_spacing_entities(text)
    t = re.sub(r"\n{3,}", "\n\n", t)

    if "Учебное заведение" not in t:
        return t

    lines = [ln.strip() for ln in t.splitlines()]
    result_lines: list[str] = []
    i = 0
    section = ""
    table_headers = {"Учебное заведение", "Winter 2026/27", "Summer 2027"}

    def _looks_like_row_start(s: str) -> bool:
        return (
            ("Universität" in s or "Technische" in s or "Fachhochschulen" in s or "University" in s)
            and not s.startswith("📌")
        )

    while i < len(lines):
        line = lines[i]
        if not line:
            result_lines.append("")
            i += 1
            continue

        if line in {"📋 Бакалавриат", "📋 Магистратура"}:
            section = line
            result_lines.append(line)
            i += 1
            continue

        if line in table_headers:
            i += 1
            continue

        if _looks_like_row_start(line):
            if i + 2 < len(lines):
                winter = lines[i + 1]
                summer = lines[i + 2]
                if winter and summer and winter not in table_headers and summer not in table_headers:
                    result_lines.append(f"📍 {line}")
                    if section:
                        result_lines.append(f"• Раздел: {section.replace('📋 ', '')}")
                    result_lines.append(f"• Winter 2026/27: {winter}")
                    result_lines.append(f"• Summer 2027: {summer}")
                    result_lines.append("")
                    i += 3
                    continue

        result_lines.append(line)
        i += 1

    out = "\n".join(result_lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _postprocess_answer_text(text: str) -> str:
    """Final answer post-processing for Telegram readability."""
    text = _normalize_spacing_entities(text)
    text = _format_deadline_tables_as_cards(text)
    return text


def _apply_confirmed_fact_guardrails(text: str, memory: dict | None) -> str:
    """Downgrade unconfirmed 'already done' claims to neutral wording."""
    if not text:
        return text
    state = (memory or {}).get("journey_state") or {}
    facts = state.get("_facts") or {}
    z_status = facts.get("zulassungsbescheid")

    # If not explicitly confirmed as done, never assert completion.
    if z_status != "done":
        text = re.sub(
            r"(?i)\bвы уже получили\s+zulassungsbescheid\b",
            "получение Zulassungsbescheid — обязательный шаг",
            text,
        )
        text = re.sub(
            r"(?i)\bу вас уже есть\s+zulassungsbescheid\b",
            "нужно получить Zulassungsbescheid",
            text,
        )
    return text


def _emit_observability_metrics(
    *,
    user_id: int,
    question: str,
    timings: dict,
    docs_retrieved: int,
    docs_graded: int,
    cache_hit: bool,
    error: str | None = None,
):
    """Emit pipeline metrics to observability backend asynchronously."""
    if not is_observability_enabled():
        return
    asyncio.create_task(asyncio.to_thread(
        log_pipeline_metrics,
        user_id=user_id,
        question=question,
        timings=timings,
        docs_retrieved=docs_retrieved,
        docs_graded=docs_graded,
        cache_hit=cache_hit,
        error=error,
    ))

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


def _is_valid_answer(answer: str) -> bool:
    """Check if answer is meaningful (not just punctuation or placeholder)."""
    if not answer:
        return False
    # Remove whitespace and check if only punctuation/ellipsis
    stripped = answer.strip()
    if not stripped:
        return False
    # Check for placeholder-like responses (just dots, ellipsis, etc.)
    if stripped in ("...", "…", ".", "..", "....", "—", "-"):
        return False
    # Must have at least some actual text (letters or digits)
    import string
    text_chars = [c for c in stripped if c not in string.punctuation and not c.isspace()]
    return len(text_chars) >= 3


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
            # Validate answer is not just ellipsis or empty
            if not _is_valid_answer(answer):
                logger.warning(f"Invalid/empty answer in JSON: '{answer[:50] if answer else ''}'")
                return (
                    "К сожалению, не удалось сформировать ответ. "
                    "Пожалуйста, переформулируйте вопрос или попробуйте позже.",
                    suggested if isinstance(suggested, list) else []
                )
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

        # Validate the extracted answer
        if not _is_valid_answer(ans):
            logger.warning(f"Invalid/empty answer after regex extraction: '{ans[:50] if ans else ''}'")
            return (
                "К сожалению, не удалось сформировать ответ. "
                "Пожалуйста, переформулируйте вопрос или попробуйте позже.",
                btns
            )

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
        if parse_mode == ParseMode.HTML and "can't parse entities" in str(e).lower():
            plain = _strip_html_to_plain(text)
            try:
                await msg.edit_text(plain or "…", reply_markup=reply_markup)
                logger.info("HTML parse fallback applied for Telegram edit")
                return True
            except Exception:
                return False
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
                    if _safe_stream_plain_enabled():
                        await _safe_edit_message(msg, _strip_html_to_plain(safe), parse_mode=None)
                    else:
                        await _safe_edit_message(msg, safe, parse_mode=ParseMode.HTML)
                last_edit_len = len(display_text)
                last_edit_time = now

    return accumulated


async def _stream_answer_with_tools(
    msg, simple_rag, question: str, context_str: str,
    chat_history: str, memory_context: str, tool_result: dict,
    prefix: str = "",
):
    """Stream LLM generation with tool result, editing the Telegram message periodically.

    Similar to _stream_answer but uses the tool-enhanced streaming method.
    Returns the full accumulated response JSON text.
    """
    accumulated = prefix
    last_edit_time = time.monotonic()
    last_edit_len = len(prefix)

    # Regex to extract partial answer from a JSON stream: {"answer": "..."}
    answer_regex = re.compile(r'"answer":\s*"((?:[^"\\]|\\.)*)', re.DOTALL)

    async for chunk in simple_rag.astream_answer_with_tools(
        question, context_str, chat_history, memory_context, get_current_date(),
        tool_result=tool_result,
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
                    if _safe_stream_plain_enabled():
                        await _safe_edit_message(msg, _strip_html_to_plain(safe), parse_mode=None)
                    else:
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


def _save_suggested(context, db_session_factory, tg_id, suggested: list, msg_id: int = None):
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
    if suggested and db_session_factory and tg_id:
        asyncio.create_task(_persist_suggested(db_session_factory, tg_id, suggested))


async def _persist_suggested(session_factory, tg_id: int, suggested: list):
    """Persist suggested questions into journey_state._last_suggested."""
    from bot.db import get_user_memory, update_user_memory
    try:
        async with session_factory() as session:
            memory = await get_user_memory(session, tg_id)
            state = memory.get("journey_state") or {}
            state["_last_suggested"] = suggested
            await update_user_memory(session, tg_id, journey_state=state)
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
    db_session_factory=None,
    use_hyde: bool = True,
    use_tools: bool = True,
    use_reranking: bool = True,
):
    """Core logic: memory → tools → rewrite → cache → retrieve → grade → rerank → stream → send.

    Enhanced with HyDE, Function Calling, and Re-ranking.
    Shared by answer(), answer_to_replied(), and handle_suggested_question().
    """
    timings = {}
    pipeline_start = time.monotonic()
    # Defaults so the except block can always reference them
    actual_question = question
    retrieved_count = 0
    doc_sources: list = []
    tool_result = None
    response_delivered = False

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

    try:
        # 1. Rewrite query for better vector search (resolves anaphora, adds keywords)
        t0 = time.monotonic()
        actual_question = await simple_rag.arewrite_query(
            question, chat_history_str, memory_context_str
        )
        timings["rewrite"] = time.monotonic() - t0

        # 1b. Check if tool/function call is needed
        if use_tools:
            t0 = time.monotonic()
            tool_needed = await simple_rag.acheck_tool_need(actual_question, memory_context_str)
            timings["tool_check"] = time.monotonic() - t0

            if tool_needed:
                t0 = time.monotonic()
                tool_result = await simple_rag.aexecute_tool(
                    tool_needed["tool_name"],
                    tool_needed["arguments"],
                )
                timings["tool_execute"] = time.monotonic() - t0
                logger.info(f"[User {tg_id}] Tool executed: {tool_needed['tool_name']}")

        # 1c. Check semantic answer cache
        t0 = time.monotonic()
        cached_response = await simple_rag.get_cached_answer(actual_question)
        timings["cache_check"] = time.monotonic() - t0

        if cached_response and not tool_result:  # Skip cache if tool was used
            clean_text, suggested = parse_suggested_buttons(cached_response)
            clean_text = _postprocess_answer_text(clean_text)
            clean_text = _apply_confirmed_fact_guardrails(clean_text, memory)
            safe_text = sanitize_telegram_html(clean_text)
            keyboard = combined_keyboard(suggested, msg.message_id) if suggested else None
            response_delivered = await _safe_edit_message(
                msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard
            )
            if suggested:
                _save_suggested(context, db_session_factory, tg_id, suggested, msg_id=msg.message_id)
            _store_qa(context, msg.message_id, question, clean_text)
            await add_chat_messages(db_session, tg_id, question, clean_text)
            if db_session_factory:
                asyncio.create_task(
                    update_journey_and_summary(simple_rag, db_session_factory, tg_id, question, clean_text)
                )
            timings["total"] = time.monotonic() - pipeline_start
            logger.info("[User %s] Cache hit | %s", tg_id, _fmt_timings(timings))
            _emit_observability_metrics(
                user_id=tg_id,
                question=question,
                timings=timings,
                docs_retrieved=0,
                docs_graded=0,
                cache_hit=True,
            )
            if db_session_factory:
                asyncio.create_task(add_pipeline_log(
                    db_session_factory, tg_id, question, actual_question,
                    cache_hit=True, docs_retrieved=0, docs_after_grading=0, timings=timings,
                ))
            return

        # 2. Retrieve candidates with HyDE and smart query routing
        await _update_processing_phase(msg, "analyze")
        t0 = time.monotonic()
        user_filters = _build_user_filters(onboarding_data)
        docs = await simple_rag.aretrieve_smart(
            actual_question,
            user_filters=user_filters,
            use_hyde=use_hyde,
            memory_context=memory_context_str,
        )
        retrieved_count = len(docs)
        timings["retrieve"] = time.monotonic() - t0

        # 3. Grade for relevance (CRAG filtering)
        t0 = time.monotonic()
        docs = await simple_rag.agrade_documents(actual_question, docs)
        timings["grade"] = time.monotonic() - t0

        # 3b. Re-rank for better precision
        if use_reranking and len(docs) > 4:
            t0 = time.monotonic()
            rerank_timeout = _get_rerank_soft_timeout_sec()
            if rerank_timeout > 0:
                try:
                    docs = await asyncio.wait_for(
                        simple_rag.arerank_documents(actual_question, docs, top_k=6),
                        timeout=rerank_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        "[User %s] Rerank soft-timeout after %.2fs, keeping graded order",
                        tg_id, rerank_timeout,
                    )
                    docs = docs[:6]
            else:
                docs = await simple_rag.arerank_documents(actual_question, docs, top_k=6)
            timings["rerank"] = time.monotonic() - t0

        # Collect source URLs from surviving docs for the pipeline log
        doc_sources = [
            doc.metadata.get("source_url", "") for doc in docs
            if doc.metadata.get("source_url")
        ]

        # Even with no docs, let LLM try to answer from system knowledge (date, greetings, etc.)
        # LLM will decide if it can answer without context based on prompt rules
        if len(docs) == 0:
            context_str = "(Контекст из базы знаний пуст — попробуй ответить из системных знаний, если вопрос общий)"
        else:
            context_str = documents_to_context_str(docs)

        # 4. Stream answer (with tool result if available)
        await _update_processing_phase(msg, "generate")
        t0 = time.monotonic()
        if tool_result:
            full_response = await _stream_answer_with_tools(
                msg, simple_rag, actual_question, context_str,
                chat_history_str, memory_context_str, tool_result,
            )
        else:
            full_response = await _stream_answer(
                msg, simple_rag, actual_question, context_str,
                chat_history_str, memory_context_str,
            )
        timings["generate"] = time.monotonic() - t0

        # Cache answer for future similar questions
        asyncio.create_task(simple_rag.cache_answer(actual_question, full_response))

        clean_text, suggested = parse_suggested_buttons(full_response)
        clean_text = _postprocess_answer_text(clean_text)
        clean_text = _apply_confirmed_fact_guardrails(clean_text, memory)

        sources_text = docs_to_sources_str(docs)
        if sources_text:
            clean_text += "\n\nИсточники/наиболее релевантные ссылки:\n" + sources_text

        safe_text = sanitize_telegram_html(clean_text)

        # Combined keyboard: suggested questions + feedback (👍/👎)
        keyboard = combined_keyboard(suggested, msg.message_id) if suggested else None
        response_delivered = await _safe_edit_message(
            msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard
        )
        if suggested:
            _save_suggested(context, db_session_factory, tg_id, suggested, msg_id=msg.message_id)

        # Store Q&A for feedback tracking
        _store_qa(context, msg.message_id, question, clean_text)

        # 5. Persist conversation + update memory
        await add_chat_messages(db_session, tg_id, question, clean_text)
        if db_session_factory:
            asyncio.create_task(
                update_journey_and_summary(simple_rag, db_session_factory, tg_id, question, clean_text)
            )

        timings["total"] = time.monotonic() - pipeline_start
        logger.info(
            "[User %s] Pipeline complete (%d docs) | %s",
            tg_id, len(docs), _fmt_timings(timings),
        )
        _emit_observability_metrics(
            user_id=tg_id,
            question=question,
            timings=timings,
            docs_retrieved=retrieved_count,
            docs_graded=len(docs),
            cache_hit=False,
        )
        if db_session_factory:
            asyncio.create_task(add_pipeline_log(
                db_session_factory, tg_id, question, actual_question,
                cache_hit=False, docs_retrieved=retrieved_count, docs_after_grading=len(docs),
                timings=timings, sources=doc_sources,
            ))

    except Exception as exc:
        timings["total"] = time.monotonic() - pipeline_start
        err_str = f"{type(exc).__name__}: {exc}"
        logger.exception("[User %s] Pipeline error: %s", tg_id, err_str)
        _emit_observability_metrics(
            user_id=tg_id,
            question=question,
            timings=timings,
            docs_retrieved=retrieved_count,
            docs_graded=0,
            cache_hit=False,
            error=err_str,
        )
        if db_session_factory:
            asyncio.create_task(add_pipeline_log(
                db_session_factory, tg_id, question, actual_question,
                cache_hit=False, docs_retrieved=retrieved_count, docs_after_grading=0,
                timings=timings, sources=doc_sources, error=err_str,
            ))
        # If the user already received the final answer, don't send a second
        # "processing error" message due to a late side-effect failure.
        if response_delivered:
            return
        raise


async def _handle_question_core_v2(
    context,
    simple_rag,
    db_session,
    chat_id: int,
    reply_to_id: int,
    tg_id: int,
    question: str,
    onboarding_data: dict,
    db_session_factory=None,
    use_hyde: bool = True,
    use_tools: bool = True,
    use_reranking: bool = True,
):
    """Pipeline-based core path (feature-flagged).

    Keeps Telegram UX/side-effects identical while delegating orchestration
    to `crag.pipeline`.
    """
    pipeline_start = time.monotonic()
    timings = {}
    response_delivered = False
    actual_question = question
    retrieved_count = 0
    doc_sources: list = []

    onboarding_data = await _ensure_onboarding_loaded(context, db_session, tg_id)
    memory, chat_history_str, memory_context_str = await _load_memory(
        db_session, tg_id, onboarding_data
    )

    msg = await context.bot.send_message(
        chat_id=chat_id,
        reply_to_message_id=reply_to_id,
        text="🔍 Ищу информацию...",
    )

    # Track if we've switched to "generate" phase
    phase_switched = {"generate": False}

    async def _stream_cb(accumulated: str):
        # Switch to "generate" phase on first content
        if not phase_switched["generate"]:
            phase_switched["generate"] = True
            await _update_processing_phase(msg, "generate")

        answer_regex = re.compile(r'"answer":\s*"((?:[^"\\]|\\.)*)', re.DOTALL)
        match = answer_regex.search(accumulated)
        if not match:
            return
        display_text = match.group(1)
        display_text = display_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        safe = sanitize_telegram_html(display_text + " ▌")
        if safe.strip():
            await _safe_edit_message(msg, safe, parse_mode=ParseMode.HTML)

    try:
        # Update phase before pipeline starts processing
        await _update_processing_phase(msg, "analyze")

        pipeline = create_default_pipeline(
            stream_callback=_stream_cb,
            use_hyde=use_hyde,
            use_decomposition=False,
            use_smart_retrieval=True,
            use_reranking=use_reranking,
            use_tools=use_tools,
            top_k=10,
            rerank_top_k=6,
        )

        ctx = PipelineContext(
            question=question,
            tg_id=tg_id,
            onboarding_data=onboarding_data,
            memory_context=memory_context_str,
            chat_history=chat_history_str,
            journey_state=memory.get("journey_state") or {},
            conversation_summary=memory.get("conversation_summary") or "",
        )

        ctx = await pipeline.run(ctx, simple_rag)
        timings = dict(ctx.timings or {})
        actual_question = ctx.rewritten_question or question
        docs = ctx.reranked_docs or ctx.graded_docs or ctx.retrieved_docs or []
        retrieved_count = len(ctx.retrieved_docs or [])
        doc_sources = [
            doc.metadata.get("source_url", "")
            for doc in docs
            if getattr(doc, "metadata", None) and doc.metadata.get("source_url")
        ]

        full_response = ctx.answer_json or json.dumps(
            {
                "answer": "К сожалению, не удалось сформировать ответ.",
                "suggested_questions": [],
            },
            ensure_ascii=False,
        )
        clean_text, suggested = parse_suggested_buttons(full_response)
        clean_text = _postprocess_answer_text(clean_text)
        clean_text = _apply_confirmed_fact_guardrails(clean_text, memory)

        sources_text = docs_to_sources_str(docs) if docs else ""
        if sources_text:
            clean_text += "\n\nИсточники/наиболее релевантные ссылки:\n" + sources_text

        safe_text = sanitize_telegram_html(clean_text)
        keyboard = combined_keyboard(suggested, msg.message_id) if suggested else None
        response_delivered = await _safe_edit_message(
            msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard
        )

        if suggested:
            _save_suggested(context, db_session_factory, tg_id, suggested, msg_id=msg.message_id)
        _store_qa(context, msg.message_id, question, clean_text)
        await add_chat_messages(db_session, tg_id, question, clean_text)
        if db_session_factory:
            asyncio.create_task(
                update_journey_and_summary(simple_rag, db_session_factory, tg_id, question, clean_text)
            )
        if not ctx.cache_hit:
            asyncio.create_task(simple_rag.cache_answer(actual_question, full_response))

        timings["total"] = time.monotonic() - pipeline_start
        logger.info(
            "[User %s] Pipeline v2 complete (%d docs) | %s",
            tg_id, len(docs), _fmt_timings(timings),
        )
        _emit_observability_metrics(
            user_id=tg_id,
            question=question,
            timings=timings,
            docs_retrieved=retrieved_count,
            docs_graded=len(ctx.graded_docs or []),
            cache_hit=bool(ctx.cache_hit),
        )
        if db_session_factory:
            asyncio.create_task(add_pipeline_log(
                db_session_factory, tg_id, question, actual_question,
                cache_hit=bool(ctx.cache_hit),
                docs_retrieved=retrieved_count,
                docs_after_grading=len(ctx.graded_docs or []),
                timings=timings, sources=doc_sources,
            ))

    except Exception as exc:
        timings["total"] = time.monotonic() - pipeline_start
        err_str = f"{type(exc).__name__}: {exc}"
        logger.exception("[User %s] Pipeline v2 error: %s", tg_id, err_str)
        _emit_observability_metrics(
            user_id=tg_id,
            question=question,
            timings=timings,
            docs_retrieved=retrieved_count,
            docs_graded=0,
            cache_hit=False,
            error=err_str,
        )
        if db_session_factory:
            asyncio.create_task(add_pipeline_log(
                db_session_factory, tg_id, question, actual_question,
                cache_hit=False, docs_retrieved=retrieved_count, docs_after_grading=0,
                timings=timings, sources=doc_sources, error=err_str,
            ))
        if response_delivered:
            return
        raise


async def _run_pipeline_v2_shadow(
    simple_rag,
    tg_id: int,
    question: str,
    onboarding_data: dict,
    db_session_factory,
    use_hyde: bool = True,
    use_tools: bool = True,
    use_reranking: bool = True,
):
    """Run pipeline v2 in background without affecting user-visible behavior."""
    if not db_session_factory:
        return

    start = time.monotonic()
    try:
        async with db_session_factory() as session:
            memory = await get_user_memory(session, tg_id)
            history_list = await get_context_messages(session, tg_id)

        chat_history_str = _build_chat_history_str(history_list)
        memory_context_str = build_memory_context(
            memory.get("journey_state"),
            memory.get("conversation_summary"),
            onboarding_data,
        )

        pipeline = create_default_pipeline(
            stream_callback=None,
            use_hyde=use_hyde,
            use_decomposition=False,
            use_smart_retrieval=True,
            use_reranking=use_reranking,
            use_tools=use_tools,
            top_k=10,
            rerank_top_k=6,
        )
        ctx = PipelineContext(
            question=question,
            tg_id=tg_id,
            onboarding_data=onboarding_data or {},
            memory_context=memory_context_str,
            chat_history=chat_history_str,
            journey_state=memory.get("journey_state") or {},
            conversation_summary=memory.get("conversation_summary") or "",
        )
        ctx = await pipeline.run(ctx, simple_rag)
        elapsed = time.monotonic() - start
        logger.info(
            "[User %s] Pipeline v2 shadow complete | total=%.2fs | cache=%s | docs=%d",
            tg_id,
            elapsed,
            bool(ctx.cache_hit),
            len(ctx.reranked_docs or ctx.graded_docs or ctx.retrieved_docs or []),
        )
    except Exception as exc:
        logger.warning(
            "[User %s] Pipeline v2 shadow failed: %s: %s",
            tg_id,
            type(exc).__name__,
            exc,
        )


async def _handle_question_with_router(
    context,
    simple_rag,
    db_session,
    chat_id: int,
    reply_to_id: int,
    tg_id: int,
    question: str,
    onboarding_data: dict,
    db_session_factory=None,
):
    """Smart routing handler that decides whether to use TOOL_ONLY, RAG_ONLY, or both.

    This is the new default handler that uses intent classification to optimize performance.
    """
    start_time = time.monotonic()

    # Step 1: Classify intent
    route = classify_intent(question)
    classification_time_ms = (time.monotonic() - start_time) * 1000

    logger.info(
        "[User %s] Intent classified: %s (confidence=%.2f, tools=%s) - %s",
        tg_id,
        route.intent.value,
        route.confidence,
        route.suggested_tools,
        route.reason,
    )

    # Log routing decision for observability
    log_routing_decision(
        user_id=tg_id,
        question=question,
        intent=route.intent.value,
        suggested_tools=route.suggested_tools,
        confidence=route.confidence,
        reason=route.reason,
        latency_ms=classification_time_ms,
    )
    increment_routing_stat(route.intent.value)

    # Step 2: Handle CHITCHAT (fastest path)
    if is_chitchat(route):
        response_text = get_chitchat_response(question)
        msg = await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_id,
            text=response_text,
        )
        _store_qa(context, msg.message_id, question, response_text)
        await add_chat_messages(db_session, tg_id, question, response_text)
        logger.info("[User %s] Chitchat response sent in %.2fs", tg_id, time.monotonic() - start_time)
        return

    # Step 3: Load user context (needed for tools and RAG)
    onboarding_data = await _ensure_onboarding_loaded(context, db_session, tg_id)
    memory, chat_history_str, memory_context_str = await _load_memory(
        db_session, tg_id, onboarding_data
    )

    tool_results = {}
    tool_context_str = ""

    # Step 4: Execute tools if needed
    if should_use_tools(route) and route.suggested_tools:
        msg = await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_id,
            text="⚙️ Проверяю данные...",
        )

        for tool_name in route.suggested_tools:
            try:
                # Execute tool with user context injection
                result = await execute_tool(
                    name=tool_name,
                    arguments={},  # LLM will provide args in full implementation
                    session_factory=db_session_factory,
                    tg_id=tg_id,
                )
                tool_results[tool_name] = result
                logger.info("[User %s] Tool %s executed: %s", tg_id, tool_name, result)
            except Exception as e:
                logger.error("[User %s] Tool %s failed: %s", tg_id, tool_name, e)
                tool_results[tool_name] = {"error": str(e)}

        # Format tool results for context
        tool_context_parts = []
        for tool_name, result in tool_results.items():
            tool_context_parts.append(f"**{tool_name}:**\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        tool_context_str = "\n\n".join(tool_context_parts)

    # Step 5: Handle TOOL_ONLY (no RAG needed)
    if route.intent == Intent.TOOL_ONLY:
        # Format tool results into readable response
        response_parts = []

        for tool_name, result in tool_results.items():
            if "error" in result:
                response_parts.append(f"❌ {result['error']}")
            elif tool_name == "get_my_progress":
                msg_text = result.get("message", "")
                stages = result.get("stages", [])
                completed = [s for s in stages if s.get("completed")]
                pending = [s for s in stages if not s.get("completed")]

                response_parts.append(f"📊 {msg_text}\n")
                if completed:
                    response_parts.append("✅ Пройдено:")
                    for s in completed:
                        response_parts.append(f"  • {s['label']}")
                if pending:
                    response_parts.append("\n⬜ Осталось:")
                    for s in pending[:5]:
                        response_parts.append(f"  • {s['label']}")

            elif tool_name == "get_next_steps":
                steps = result.get("next_steps", [])
                if steps:
                    response_parts.append("🎯 Рекомендую сделать дальше:\n")
                    for i, step in enumerate(steps, 1):
                        priority_emoji = {"urgent": "🔥", "high": "⚡", "medium": "📌"}.get(
                            step.get("priority", "medium"), "📌"
                        )
                        response_parts.append(f"{i}. {priority_emoji} {step.get('action', step.get('stage', ''))}")

            elif tool_name in ("check_deadline", "calculate_budget", "calculate_days_until"):
                # Simple JSON dump for now
                response_parts.append(json.dumps(result, ensure_ascii=False, indent=2))

            else:
                response_parts.append(result.get("message", json.dumps(result, ensure_ascii=False)))

        final_response = "\n\n".join(response_parts) if response_parts else "Не удалось получить данные."

        # Update or send new message
        safe_text = sanitize_telegram_html(final_response)
        if 'msg' in locals():
            await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML)
        else:
            msg = await context.bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=reply_to_id,
                text=safe_text,
                parse_mode=ParseMode.HTML,
            )

        _store_qa(context, msg.message_id, question, final_response)
        await add_chat_messages(db_session, tg_id, question, final_response)

        logger.info(
            "[User %s] TOOL_ONLY response complete in %.2fs",
            tg_id,
            time.monotonic() - start_time
        )
        return

    # Step 6: Use RAG (for RAG_ONLY or TOOL_THEN_RAG)
    # Delegate to existing pipeline with tool context
    if not 'msg' in locals():
        msg = await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_id,
            text="🔍 Ищу информацию...",
        )

    # Build enhanced memory context with tool results
    enhanced_memory = memory_context_str
    if tool_context_str:
        enhanced_memory += f"\n\n## Данные из персональных инструментов:\n{tool_context_str}"

    # Now call the v2 pipeline with enhanced context
    # (This is a simplified version - in production you'd want to stream, etc.)
    await _update_processing_phase(msg, "analyze")

    try:
        pipeline = create_default_pipeline(
            stream_callback=None,  # Simplified for now
            use_hyde=True,
            use_decomposition=False,
            use_smart_retrieval=True,
            use_reranking=True,
            use_tools=False,  # We already executed tools above
            top_k=10,
            rerank_top_k=6,
        )

        ctx = PipelineContext(
            question=question,
            tg_id=tg_id,
            onboarding_data=onboarding_data,
            memory_context=enhanced_memory,
            chat_history=chat_history_str,
            journey_state=memory.get("journey_state") or {},
            conversation_summary=memory.get("conversation_summary") or "",
        )

        ctx = await pipeline.run(ctx, simple_rag)

        full_response = ctx.answer_json or json.dumps(
            {"answer": "К сожалению, не удалось сформировать ответ.", "suggested_questions": []},
            ensure_ascii=False,
        )

        clean_text, suggested = parse_suggested_buttons(full_response)
        clean_text = _postprocess_answer_text(clean_text)
        clean_text = _apply_confirmed_fact_guardrails(clean_text, memory)

        docs = ctx.reranked_docs or ctx.graded_docs or ctx.retrieved_docs or []
        sources_text = docs_to_sources_str(docs) if docs else ""
        if sources_text:
            clean_text += "\n\nИсточники:\n" + sources_text

        safe_text = sanitize_telegram_html(clean_text)
        keyboard = combined_keyboard(suggested, msg.message_id) if suggested else None

        await _safe_edit_message(msg, safe_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

        if suggested:
            _save_suggested(context, db_session_factory, tg_id, suggested, msg_id=msg.message_id)
        _store_qa(context, msg.message_id, question, clean_text)
        await add_chat_messages(db_session, tg_id, question, clean_text)

        if db_session_factory:
            asyncio.create_task(
                update_journey_and_summary(simple_rag, db_session_factory, tg_id, question, clean_text)
            )

        logger.info(
            "[User %s] %s response complete in %.2fs | %d docs",
            tg_id,
            route.intent.value,
            time.monotonic() - start_time,
            len(docs),
        )

    except Exception as exc:
        logger.exception("[User %s] RAG pipeline error", tg_id)
        await _safe_edit_message(
            msg,
            "⚠️ Произошла ошибка при обработке запроса. Попробуй переформулировать вопрос.",
            parse_mode=None,
        )


async def _handle_question_with_optional_trace(handler_coro, tg_id: int, question: str):
    """Execute handler with optional observability trace context."""
    if is_observability_enabled():
        async with trace_full_pipeline(
            user_id=tg_id,
            question=question,
            metadata={"path": "telegram_handler"},
        ):
            return await handler_coro
    return await handler_coro


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
        use_routing = _use_routing_enabled()
        use_new = _use_new_pipeline_enabled()

        # Priority: routing > new pipeline > legacy
        if use_routing:
            handler = _handle_question_with_router
        elif use_new:
            handler = _handle_question_core_v2
        else:
            handler = _handle_question_core

        user_id = update.effective_user.id
        await _handle_question_with_optional_trace(
            handler(
                context=context,
                simple_rag=simple_rag,
                db_session=db_session,
                chat_id=update.effective_chat.id,
                reply_to_id=update.effective_message.id,
                tg_id=user_id,
                question=question,
                onboarding_data=context.user_data.get("onboarding", {}),
                db_session_factory=kwargs.get("db_session_factory"),
            ),
            tg_id=user_id,
            question=question,
        )
        if not use_new and _use_pipeline_shadow_enabled():
            asyncio.create_task(
                _run_pipeline_v2_shadow(
                    simple_rag=simple_rag,
                    tg_id=user_id,
                    question=question,
                    onboarding_data=dict(context.user_data.get("onboarding", {})),
                    db_session_factory=kwargs.get("db_session_factory"),
                )
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
        use_new = _use_new_pipeline_enabled()
        handler = _handle_question_core_v2 if use_new else _handle_question_core
        user_id = update.effective_user.id
        await _handle_question_with_optional_trace(
            handler(
                context=context,
                simple_rag=simple_rag,
                db_session=db_session,
                chat_id=update.effective_chat.id,
                reply_to_id=update.effective_message.reply_to_message.id,
                tg_id=user_id,
                question=question,
                onboarding_data=context.user_data.get("onboarding", {}),
                db_session_factory=kwargs.get("db_session_factory"),
            ),
            tg_id=user_id,
            question=question,
        )
        if not use_new and _use_pipeline_shadow_enabled():
            asyncio.create_task(
                _run_pipeline_v2_shadow(
                    simple_rag=simple_rag,
                    tg_id=user_id,
                    question=question,
                    onboarding_data=dict(context.user_data.get("onboarding", {})),
                    db_session_factory=kwargs.get("db_session_factory"),
                )
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
