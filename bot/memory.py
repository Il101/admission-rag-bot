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

from sqlalchemy.ext.asyncio import AsyncSession
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


def _truncate_summary(summary: str, max_words: int = 250) -> str:
    """Keep summary under max_words by trimming oldest sentences."""
    words = summary.split()
    if len(words) <= max_words:
        return summary
    # Keep the last max_words words (most recent info)
    return "... " + " ".join(words[-max_words:])


async def update_journey_and_summary(
    simple_rag,
    session: AsyncSession,
    tg_id: int,
    question: str,
    response: str,
) -> None:
    """Update journey state, conversation summary, and mood after a RAG response.

    Uses fast keyword-based classification instead of an LLM call.
    Runs asynchronously — should be awaited after sending the response to the user.
    """
    try:
        memory = await get_user_memory(session, tg_id)
        current_state = memory["journey_state"] or DEFAULT_JOURNEY_STATE.copy()
        current_summary = memory["conversation_summary"] or "Начало общения."

        # 1. Detect discussed stages from question + response
        combined_text = f"{question} {response}"
        newly_discussed = _detect_stages(combined_text)

        new_state = dict(current_state)
        for stage_id in newly_discussed:
            new_state[stage_id] = "discussed"

        # Never downgrade "discussed" back to "pending"
        for stage in JOURNEY_STAGES:
            if current_state.get(stage) == "discussed":
                new_state[stage] = "discussed"

        # 2. Detect mood
        user_mood = _detect_mood(question)
        new_state["_user_mood"] = user_mood

        # 3. Append to summary + periodic LLM compression
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
        # Non-critical — bot continues working without memory update


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

