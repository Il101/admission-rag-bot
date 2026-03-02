"""
Hybrid Memory Manager for the admission bot.

Three layers:
1. Journey State (JSONB) — which admission stages have been discussed + user mood
2. Conversation Summary (TEXT) — compressed history of all past interactions + key facts
3. Recent Messages (existing) — last 6 raw messages
"""

import json
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


# ─── Memory Update via LLM ───────────────────────────────────────

MEMORY_UPDATE_PROMPT = """Ты — система обновления памяти бота-проводника по поступлению в Австрию.

Тебе даны:
1. Текущий journey_state — какие этапы пути обсуждались
2. Текущий conversation_summary — краткое описание всего, что обсуждалось ранее
3. Новый вопрос пользователя и ответ бота

Обнови ВСЕ поля:

1. journey_state: отметь как "discussed" этапы, которые затронуты в новом вопросе/ответе.

2. conversation_summary: добавь ключевые факты из нового обмена, сохрани старую информацию, не превышай 250 слов. ВАЖНО: включай конкретные факты о пользователе, если он их упоминает (название вуза, программа, статус документов, города, сроки).

3. user_mood: определи текущее эмоциональное состояние пользователя по его вопросу. Одно из: "calm", "anxious", "confused", "frustrated", "excited".

ЭТАПЫ ПУТИ:
documents_prep — подготовка документов (аттестат, апостиль, перевод)
nostrification — нострификация (признание документов)
language — языковые требования (немецкий, английский, сертификаты)
university_choice — выбор университета и программы
application — процесс подачи заявки
admission_letter — получение письма о зачислении
visa_residence — виза, ВНЖ, Aufenthaltstitel
finances — стоимость, стипендии, финансирование
housing_relocation — жильё, переезд, быт

Ответь СТРОГО в формате JSON без markdown:
{{"journey_state": {{"documents_prep": "pending|discussed"}}, "summary": "обновлённый текст", "user_mood": "calm"}}

Текущий journey_state: {current_state}

Текущий summary: {current_summary}

Новый вопрос: {question}

Ответ бота: {response}"""


async def update_journey_and_summary(
    simple_rag,
    session: AsyncSession,
    tg_id: int,
    question: str,
    response: str,
) -> None:
    """Update journey state, conversation summary, and mood after a RAG response.
    
    Runs asynchronously — should be awaited after sending the response to the user.
    """
    try:
        memory = await get_user_memory(session, tg_id)
        current_state = memory["journey_state"] or DEFAULT_JOURNEY_STATE.copy()
        current_summary = memory["conversation_summary"] or "Начало общения."

        prompt = MEMORY_UPDATE_PROMPT.format(
            current_state=json.dumps(current_state, ensure_ascii=False),
            current_summary=current_summary,
            question=question[:500],
            response=response[:1000]
        )
        
        result = await simple_rag.agenerate_json(prompt)

        # Parse the JSON response
        # Strip markdown code fences if present
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1] if "\n" in result else result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()

        parsed = json.loads(result)
        new_state = parsed.get("journey_state", current_state)
        new_summary = parsed.get("summary", current_summary)
        user_mood = parsed.get("user_mood", "calm")

        # Merge: never downgrade "discussed" back to "pending"
        for stage in JOURNEY_STAGES:
            if current_state.get(stage) == "discussed":
                new_state[stage] = "discussed"

        # Store mood as a special key in journey_state
        new_state["_user_mood"] = user_mood

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

