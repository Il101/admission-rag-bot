"""
Hybrid Memory Manager for the admission bot.

Three layers:
1. Journey State (JSONB) — which admission stages have been discussed + user mood
2. Conversation Summary (TEXT) — compressed history of all past interactions + key facts
3. Recent Messages (existing) — last 6 raw messages
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from bot.db import get_user_memory, update_user_memory

logger = logging.getLogger(__name__)


# ─── Journey Stages ──────────────────────────────────────────────

JOURNEY_STAGES = {
    "documents_prep": {
        "label": "📄 Подготовка документов",
        "question": "Какие документы нужно подготовить?",
        "prerequisites": [],
    },
    "nostrification": {
        "label": "📜 Нострификация",
        "question": "Нужна ли мне нострификация?",
        "prerequisites": ["documents_prep"],
    },
    "language": {
        "label": "🗣 Языковой сертификат",
        "question": "Какой уровень немецкого нужен?",
        "prerequisites": [],
    },
    "university_choice": {
        "label": "🏫 Выбор университета",
        "question": "Какие университеты мне подходят?",
        "prerequisites": [],
    },
    "application": {
        "label": "📝 Подача заявки",
        "question": "Как подать заявку на поступление?",
        "prerequisites": ["documents_prep", "language"],
    },
    "admission_letter": {
        "label": "✉️ Зачисление",
        "question": "Как получить письмо о зачислении?",
        "prerequisites": ["application"],
    },
    "visa_residence": {
        "label": "🛂 Виза и ВНЖ",
        "question": "Как получить ВНЖ студента?",
        "prerequisites": ["documents_prep", "admission_letter"],
    },
    "finances": {
        "label": "💰 Финансы и стипендии",
        "question": "Сколько стоит обучение?",
        "prerequisites": [],
    },
    "housing_relocation": {
        "label": "🏠 Жильё и переезд",
        "question": "Как найти жильё в Австрии?",
        "prerequisites": [],
    },
}

DEFAULT_JOURNEY_STATE = {stage: "pending" for stage in JOURNEY_STAGES}


# ─── Keyword-based Memory Update (replaces LLM call) ─────────────

# Keywords that indicate a journey stage was discussed
STAGE_KEYWORDS: dict[str, list[str]] = {
    "documents_prep": [
        "документ", "аттестат", "апостиль", "перевод", "справк", "диплом",
        "нотариальн", "легализ", "копи", "оригинал", "выписк", "приложени",
    ],
    "nostrification": [
        "нострификац", "признание", "anerkenn", "nostrifik", "bewertung",
        "подтвержден", "эквивалент",
    ],
    "language": [
        "немецк", "английск", "язык", "сертификат", "deutsch", "goethe",
        "osd", "testdaf", "ielts", "toefl", "dsh", "a1", "a2", "b1", "b2",
        "c1", "c2", "ergänzungsprüf", "vorstudienlehrgang", "курс",
    ],
    "university_choice": [
        "университет", "вуз", "uni ", "fachhochschule", "программ",
        "специальност", "факультет", "tu wien", "uni wien", "uni graz",
        "uni innsbruck", "boku", "wu wien", "выбор", "рейтинг",
    ],
    "application": [
        "подача", "заявк", "application", "zulassung", "bewerbung",
        "дедлайн", "срок подач", "онлайн заявк", "подать", "apply",
    ],
    "admission_letter": [
        "зачислен", "зачисл", "письмо", "zulassungsbescheid", "admission",
        "offer", "приём", "принят",
    ],
    "visa_residence": [
        "виз", "внж", "вид на жительство", "aufenthalts", "residence",
        "permit", "ma35", "магистрат", "посольств", "консульств",
        "страховк", "финансов", "гарант",
    ],
    "finances": [
        "стоимост", "стипенди", "оплат", "studiengebühr", "взнос",
        "scholarship", "oead", "грант", "бюджет", "расход", "сколько стоит",
        "финанс", "банк",
    ],
    "housing_relocation": [
        "жильё", "жилье", "квартир", "общежити", "studentenheim", "wg ",
        "wohnung", "miet", "переезд", "аренд", "регистрац", "мельдунг",
        "meldezettel",
    ],
}

# Mood keywords
MOOD_KEYWORDS: dict[str, list[str]] = {
    "anxious": ["боюсь", "волнуюсь", "страшно", "паник", "не успе", "тревож"],
    "confused": ["запутал", "не понимаю", "сложно", "непонятно", "столько всего"],
    "frustrated": ["достало", "опять", "почему так", "бесит", "устал", "надоел"],
    "excited": ["круто", "ура", "класс", "здорово", "отлично", "супер"],
}


# ── Entity Extraction Patterns ────────────────────────────────────────────

import re

ENTITY_PATTERNS: dict[str, list[str]] = {
    "university": [
        r"(?i)\b(uni(?:versität)?|университет)\s+(wien|вена|graz|грац|innsbruck|инсбрук|salzburg|зальцбург|linz|линц)",
        r"(?i)\b(tu|technische)\s+(wien|graz)",
        r"(?i)\bboku\b",
        r"(?i)\bwu\s+wien\b",
        r"(?i)\bmeduni\s+wien\b",
        r"(?i)\bjku\s+linz\b",
        r"(?i)\bfh\s+\w+\b",
    ],
    "deadline": [
        r"(\d{1,2})[./](\d{1,2})[./](20\d{2})",
        r"до\s+(\d{1,2})\s+(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)",
        r"дедлайн[а-я]*\s+(\d{1,2}[./]\d{1,2}[./]?\d{0,4})",
    ],
    "document": [
        r"(?i)\b(аттестат|диплом|апостиль|нострификаци[яю]|справк[ау]|сертификат)\b",
        r"(?i)\b(перевод|transcript|zulassungsbescheid)\b",
    ],
    "program": [
        r"(?i)\b(bachelor|бакалавр|master|магистр|phd|докторант)\b",
        r"(?i)\b(informatik|информатик|wirtschaft|экономик|medizin|медицин)\b",
    ],
    "city": [
        r"(?i)\b(вен[аеу]|vienna|wien)\b",
        r"(?i)\b(грац[е]?|graz)\b",
        r"(?i)\b(инсбрук[е]?|innsbruck)\b",
        r"(?i)\b(линц[е]?|linz)\b",
        r"(?i)\b(зальцбург[е]?|salzburg)\b",
    ],
    "language_level": [
        r"(?i)\b([ABC][12])\b",
        r"(?i)\b(goethe|ösd|osd|testdaf|dsh|ielts|toefl)\b",
    ],
}


def extract_entities(text: str) -> list[dict]:
    """Extract structured entities from text using regex patterns.

    Returns a list of dicts with keys: type, value, source
    """
    entities = []
    text_lower = text.lower()

    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        value = " ".join(m for m in match if m).strip()
                    else:
                        value = match.strip()

                    if value and len(value) > 1:
                        # Normalize common values
                        value = _normalize_entity_value(entity_type, value)
                        if value:
                            entities.append({
                                "type": entity_type,
                                "value": value,
                                "source": "extracted",
                            })
            except re.error:
                continue

    # Deduplicate
    seen = set()
    unique_entities = []
    for e in entities:
        key = (e["type"], e["value"].lower())
        if key not in seen:
            seen.add(key)
            unique_entities.append(e)

    return unique_entities


def _normalize_entity_value(entity_type: str, value: str) -> str:
    """Normalize entity values for consistency."""
    value = value.strip()

    if entity_type == "university":
        # Normalize university names
        normalizations = {
            "universität wien": "Uni Wien",
            "uni wien": "Uni Wien",
            "университет вена": "Uni Wien",
            "tu wien": "TU Wien",
            "technische wien": "TU Wien",
            "tu graz": "TU Graz",
            "technische graz": "TU Graz",
            "wu wien": "WU Wien",
            "meduni wien": "MedUni Wien",
            "jku linz": "JKU Linz",
            "boku": "BOKU Wien",
            "uni graz": "Uni Graz",
            "universität graz": "Uni Graz",
            "университет грац": "Uni Graz",
            "uni innsbruck": "Uni Innsbruck",
            "universität innsbruck": "Uni Innsbruck",
            "университет инсбрук": "Uni Innsbruck",
            "uni salzburg": "Uni Salzburg",
            "universität salzburg": "Uni Salzburg",
            "университет зальцбург": "Uni Salzburg",
        }
        return normalizations.get(value.lower(), value)

    elif entity_type == "city":
        normalizations = {
            "вена": "Vienna", "вене": "Vienna", "вену": "Vienna",
            "vienna": "Vienna", "wien": "Vienna",
            "грац": "Graz", "граце": "Graz", "graz": "Graz",
            "инсбрук": "Innsbruck", "инсбруке": "Innsbruck", "innsbruck": "Innsbruck",
            "линц": "Linz", "линце": "Linz", "linz": "Linz",
            "зальцбург": "Salzburg", "зальцбурге": "Salzburg", "salzburg": "Salzburg",
        }
        return normalizations.get(value.lower(), value)

    elif entity_type == "program":
        normalizations = {
            "bachelor": "bachelor", "бакалавр": "bachelor",
            "master": "master", "магистр": "master",
            "phd": "phd", "докторант": "phd",
        }
        return normalizations.get(value.lower(), value.lower())

    elif entity_type == "language_level":
        # Normalize to uppercase
        return value.upper()

    return value


async def extract_and_store_entities(
    session_factory,
    tg_id: int,
    text: str,
) -> list[dict]:
    """Extract entities from text and store them in the database.

    Returns the list of extracted entities.
    """
    from bot.db import add_user_entity

    entities = extract_entities(text)

    if entities:
        async with session_factory() as session:
            for entity in entities:
                await add_user_entity(
                    session,
                    tg_id=tg_id,
                    entity_type=entity["type"],
                    entity_value=entity["value"],
                    source=entity["source"],
                )

    return entities


def _detect_stages(text: str) -> list[str]:
    """Detect which journey stages are mentioned in the given text."""
    text_lower = text.lower()
    detected = []
    for stage_id, keywords in STAGE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                detected.append(stage_id)
                break
    return detected


def _detect_mood(question: str) -> str:
    """Detect user mood from their question using keyword matching."""
    q_lower = question.lower()
    for mood, keywords in MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                return mood
    return "calm"


def _extract_fact_updates(question: str) -> dict[str, str]:
    """Extract explicit user-confirmed facts from the user message.

    Returns a sparse map, e.g. {"zulassungsbescheid": "done"|"not_done"}.
    """
    text = (question or "").lower()
    updates: dict[str, str] = {}

    if "zulassungsbescheid" in text:
        not_done_markers = [
            "не получил",
            "не получал",
            "не получала",
            "еще не получил",
            "ещё не получил",
            "еще не получал",
            "ещё не получал",
            "еще не получала",
            "ещё не получала",
            "пока не получил",
            "пока не получал",
            "пока не получала",
            "нет zulassungsbescheid",
            "zulassungsbescheid нет",
            "не получен",
        ]
        done_markers = [
            "получил",
            "получила",
            "получен",
            "уже есть",
            "есть zulassungsbescheid",
            "имею zulassungsbescheid",
            "на руках zulassungsbescheid",
        ]

        if any(m in text for m in not_done_markers):
            updates["zulassungsbescheid"] = "not_done"
        elif any(m in text for m in done_markers):
            updates["zulassungsbescheid"] = "done"

    return updates


def _truncate_summary(summary: str, max_words: int = 250) -> str:
    """Keep summary under max_words by trimming oldest sentences."""
    words = summary.split()
    if len(words) <= max_words:
        return summary
    # Keep the last max_words words (most recent info)
    return "... " + " ".join(words[-max_words:])


async def update_journey_and_summary(
    simple_rag,
    session_factory,
    tg_id: int,
    question: str,
    response: str,
) -> None:
    """Update journey state, conversation summary, and mood after a RAG response.

    Uses fast keyword-based classification instead of an LLM call.
    Opens its own DB session so it can safely run as a fire-and-forget task.
    """
    try:
        async with session_factory() as session:
            memory = await get_user_memory(session, tg_id)
            current_state = memory["journey_state"] or DEFAULT_JOURNEY_STATE.copy()
            current_summary = memory["conversation_summary"] or "Начало общения."

            # 1. Detect discussed stages from USER message only.
            # This prevents "self-confirmation" where bot output mutates memory.
            newly_discussed = _detect_stages(question)

            new_state = dict(current_state)
            for stage_id in newly_discussed:
                new_state[stage_id] = "discussed"

            # Never downgrade "discussed" back to "pending"
            for stage in JOURNEY_STAGES:
                if current_state.get(stage) == "discussed":
                    new_state[stage] = "discussed"

            # 2. Update explicit user-confirmed facts
            fact_updates = _extract_fact_updates(question)
            if fact_updates:
                current_facts = dict(current_state.get("_facts") or {})
                current_facts.update(fact_updates)
                new_state["_facts"] = current_facts

            # 3. Detect mood
            user_mood = _detect_mood(question)
            new_state["_user_mood"] = user_mood

            # 4. Append to summary + periodic LLM compression
            q_snippet = question[:150].strip()
            r_snippet = response[:200].strip()
            new_entry = f"В: {q_snippet} | О: {r_snippet}"
            new_summary = f"{current_summary}\n{new_entry}"

            # Compress with LLM every 5 exchanges to keep summary focused
            exchange_count = new_summary.count("\nВ: ")
            if exchange_count >= 5 and exchange_count % 5 == 0 and simple_rag is not None:
                try:
                    new_summary = await simple_rag.acompress_summary(new_summary)
                except Exception:
                    logger.warning("LLM summary compression failed, using truncation")
                    new_summary = _truncate_summary(new_summary)
            else:
                new_summary = _truncate_summary(new_summary)

            await update_user_memory(session, tg_id, new_state, new_summary)

    except Exception:
        logger.exception("Failed to update journey/summary memory")


# ─── Build Memory Context for Prompt ──────────────────────────────

def _get_user_stage(journey_state: Optional[dict]) -> tuple:
    """Determine user's stage and count of discussed topics."""
    if not journey_state:
        return "novice", 0

    discussed_count = sum(
        1 for stage_id, status in journey_state.items()
        if not stage_id.startswith("_") and status == "discussed"
    )

    if discussed_count <= 2:
        return "novice", discussed_count
    elif discussed_count <= 6:
        return "in_progress", discussed_count
    else:
        return "advanced", discussed_count


def _get_skipped_step_warnings(journey_state: Optional[dict]) -> list:
    """Detect if user is discussing advanced topics but skipped prerequisites."""
    if not journey_state:
        return []

    warnings = []
    for stage_id, status in journey_state.items():
        if stage_id.startswith("_") or status != "discussed":
            continue
        stage_info = JOURNEY_STAGES.get(stage_id)
        if not stage_info:
            continue
        for prereq_id in stage_info.get("prerequisites", []):
            if journey_state.get(prereq_id) != "discussed":
                prereq_info = JOURNEY_STAGES.get(prereq_id, {})
                warnings.append(
                    f"⚠️ Пользователь обсуждал «{stage_info['label']}», "
                    f"но НЕ обсуждал предшествующий этап «{prereq_info.get('label', prereq_id)}». "
                    f"Мягко уточни, не пропустил ли он этот шаг."
                )
    return warnings


def _get_current_date_str() -> str:
    """Get current date formatted for the prompt."""
    # CET/CEST timezone (Austria)
    cet = timezone(timedelta(hours=1))
    now = datetime.now(cet)
    months_ru = {
        1: "января", 2: "февраля", 3: "марта", 4: "апреля",
        5: "мая", 6: "июня", 7: "июля", 8: "августа",
        9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
    }
    return f"{now.day} {months_ru[now.month]} {now.year}"


def build_memory_context(
    journey_state: Optional[dict],
    conversation_summary: Optional[str],
    onboarding_data: Optional[dict] = None,
) -> str:
    """Build a memory context string to inject into the prompt.
    
    Now includes: user profile, stage detection, adaptive guidance,
    skipped-step warnings, mood, and discussed/pending topics.
    """
    parts = []

    # User profile from onboarding
    if onboarding_data:
        profile_parts = []
        if onboarding_data.get("countryScope"):
            profile_parts.append(f"Страна: {onboarding_data['countryScope']}")
        if onboarding_data.get("document"):
            profile_parts.append(f"Документ: {onboarding_data['document']}")
        if onboarding_data.get("targetLevel"):
            profile_parts.append(f"Цель: {onboarding_data['targetLevel']}")
        if onboarding_data.get("germanLevel"):
            profile_parts.append(f"Немецкий: {onboarding_data['germanLevel']}")
        if onboarding_data.get("englishLevel"):
            profile_parts.append(f"Английский: {onboarding_data['englishLevel']}")
        if profile_parts:
            parts.append("👤 Профиль: " + ", ".join(profile_parts))

    # Stage detection and adaptive guidance
    stage, discussed_count = _get_user_stage(journey_state)
    total_stages = len(JOURNEY_STAGES)

    if stage == "novice":
        parts.append(
            f"📊 Стадия: НОВИЧОК ({discussed_count}/{total_stages} этапов обсуждено)\n"
            "🎯 Инструкция: Объясняй простыми словами. Австрийские термины — всегда с переводом. "
            "Давай общую картину. Не перегружай деталями."
        )
    elif stage == "in_progress":
        parts.append(
            f"📊 Стадия: В ПРОЦЕССЕ ({discussed_count}/{total_stages} этапов обсуждено)\n"
            "🎯 Инструкция: Будь конкретнее. Термины можешь использовать без пояснений. "
            "Фокусируйся на сроках и следующих шагах."
        )
    else:
        parts.append(
            f"📊 Стадия: ПРОДВИНУТЫЙ ({discussed_count}/{total_stages} этапов обсуждено)\n"
            "🎯 Инструкция: Максимально лаконично. Не повторяй базу. Давай нюансы и детали."
        )

    # User mood
    if journey_state:
        mood = journey_state.get("_user_mood", "calm")
        mood_labels = {
            "calm": "😊 спокойное",
            "anxious": "😟 тревожное — ПОДДЕРЖИ перед ответом",
            "confused": "😕 растерянное — УПРОСТИ ответ",
            "frustrated": "😤 раздражённое — ПРИЗНАЙ сложность и дай конкретику",
            "excited": "🤩 воодушевлённое",
        }
        mood_label = mood_labels.get(mood, mood_labels["calm"])
        parts.append(f"💭 Настроение: {mood_label}")

    # Journey state — discussed vs pending
    if journey_state:
        discussed = []
        pending = []
        for stage_id, status in journey_state.items():
            if stage_id.startswith("_"):
                continue
            stage_info = JOURNEY_STAGES.get(stage_id)
            if not stage_info:
                continue
            if status == "discussed":
                discussed.append(f"✅ {stage_info['label']}")
            else:
                pending.append(f"⬜ {stage_info['label']}")
        if discussed:
            parts.append("Уже обсуждали:\n" + "\n".join(discussed))
        if pending:
            parts.append("Ещё не обсуждали:\n" + "\n".join(pending))

        facts = journey_state.get("_facts") or {}
        fact_lines = []
        z_status = facts.get("zulassungsbescheid")
        if z_status == "done":
            fact_lines.append("✅ Zulassungsbescheid: подтверждено пользователем как получено")
        elif z_status == "not_done":
            fact_lines.append("⬜ Zulassungsbescheid: пользователь явно сказал, что еще не получен")
        if fact_lines:
            parts.append("Подтвержденные факты пользователя:\n" + "\n".join(fact_lines))

    # Skipped step warnings
    warnings = _get_skipped_step_warnings(journey_state)
    if warnings:
        parts.append("\n".join(warnings))

    # Conversation summary
    if conversation_summary:
        parts.append(f"Краткое содержание предыдущих разговоров:\n{conversation_summary}")

    return "\n\n".join(parts) if parts else "Новый пользователь, информации пока нет."


def get_current_date() -> str:
    """Get the current date string for injection into the prompt."""
    return _get_current_date_str()


# ─── Fallback Buttons ─────────────────────────────────────────────

def get_fallback_buttons(journey_state: Optional[dict]) -> list:
    """Generate suggested question buttons from undiscussed journey stages."""
    if not journey_state:
        # New user — offer initial questions
        return [
            "📋 Дай мне чек-лист для поступления",
            "🏫 Какие университеты мне подходят?",
            "📄 Какие документы нужно подготовить?",
        ]

    pending = []
    for stage_id, status in journey_state.items():
        if stage_id.startswith("_"):
            continue
        if status != "discussed":
            stage_info = JOURNEY_STAGES.get(stage_id)
            if stage_info:
                pending.append(stage_info["question"])

    if not pending:
        # All stages discussed!
        return [
            "📋 Обнови мой чек-лист",
            "❓ У меня ещё вопрос",
        ]

    return pending[:3]
