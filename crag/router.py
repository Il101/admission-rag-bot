"""
Intent Router for the admission bot.

Classifies user queries to determine the best handling strategy:
- TOOL_ONLY: Answer using tools alone, no RAG needed
- RAG_ONLY: Use knowledge base search
- TOOL_THEN_RAG: Use tools for data, then RAG for context
- CHITCHAT: Simple conversational response
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Query intent categories."""
    TOOL_ONLY = "tool_only"        # Can answer with tools alone
    RAG_ONLY = "rag_only"          # Needs knowledge base search
    TOOL_THEN_RAG = "tool_rag"     # Tools + RAG for complete answer
    CHITCHAT = "chitchat"          # Greeting, thanks, etc.


@dataclass
class RouteResult:
    """Result of intent classification."""
    intent: Intent
    suggested_tools: List[str]
    confidence: float
    reason: str


# ── Keyword Patterns for Intent Detection ──────────────────────────────


# Personal progress patterns (TOOL_ONLY: get_my_progress, get_next_steps)
PERSONAL_PROGRESS_PATTERNS = [
    r"(мой|моя|мои|моё)\s*(прогресс|статус|этап|чек-?лист|профиль)",
    r"на каком.*(я|этапе|шаге)",
    r"где я (сейчас|нахожусь)",
    r"что (я|мне).*(сделал|прошел|прошёл)",
    r"что (дальше|следующ|делать)",
    r"(с чего|как) начать",
    r"следующ(ий|ие|ая) шаг",
    r"что (ты )?(знаешь|помнишь) (обо |про )?мн",
    r"мо(й|и|я) (данные|информаци|универ|город)",
]

# Deadline/budget calculation patterns (TOOL_ONLY: check_deadline, calculate_budget)
CALCULATOR_PATTERNS = [
    r"(дедлайн|срок|deadline).*(подач|когда|для|на)",
    r"когда.*(подавать|подать|дедлайн|срок)",
    r"(сколько|какой).*(стоит|бюджет|расход|денег|нужно денег)",
    r"(бюджет|расход|стоимость).*(на|в|для)",
    r"(рассчита|посчита).*(бюджет|расход|стоимость)",
    r"сколько (дней|времени) (до|осталось)",
    r"успе(ю|ваю|ть).*(подать|дедлайн)",
    r"дедлайн.*(bachelor|бакалавр|master|магистр|phd)",
]

# Document checklist patterns (TOOL_ONLY: get_document_checklist)
DOCUMENT_PATTERNS = [
    r"(как(ие|ой)|список|чек-?лист).*(документ|бумаг)",
    r"документ.*(нужн|нужен|необходим|требу)",
    r"(что|какие).*(собрать|подготовить|нужн).*(документ|бумаг)",
]

# Knowledge base patterns (RAG_ONLY)
KNOWLEDGE_PATTERNS = [
    r"(что такое|как работает|как устроен|объясни|расскажи)",
    r"(нострификаци|апостиль|aufnahme|ergänzungsprüf|vorstudienlehrgang)",
    r"(как|где).*(получить|оформить|сделать|подать|зарегистрироват)",
    r"(чем|в чём|какая) (разниц|отличи|различи)",
    r"(можно ли|разрешено|допускается)",
    r"(требовани|условия|критерии).*(для|к|на)",
    r"(процесс|процедура|порядок|этапы).*(поступлени|подач|получени)",
    r"(универ|вуз|программ).*(подходит|лучше|рейтинг|сравни)",
    r"(виз|внж|вид на жительство|aufenthalt)",
    r"(стипенди|грант|финансирован|scholarship)",
    r"(жильё|жилье|общежити|studentenheim|wg|квартир)",
    r"(страховк|versicherung|ögk)",
    r"(язык|немецк|deutsch|goethe|ösd|testdaf)",
]

# Chitchat patterns
CHITCHAT_PATTERNS = [
    r"^(привет|здравствуй|добр|хай|хей|hello|hi|hey)[\s!.]*$",
    r"^(спасибо|благодар|thanks|thank you|данке|danke).*[\s!.]*$",
    r"^благодарю[\s!.]*$",
    r"^(пока|до свидания|bye|чао|ciao)[\s!.]*$",
    r"^(как (дела|ты|сам)|что нового)[\s?]*$",
    r"^(ок|окей|okay|понял|понятно|ясно|good|хорошо|отлично)[\s!.]*$",
]


def _match_patterns(text: str, patterns: List[str]) -> bool:
    """Check if text matches any of the patterns."""
    text_lower = text.lower().strip()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def _get_matching_patterns(text: str, patterns: List[str]) -> List[str]:
    """Return all matching patterns for debugging."""
    text_lower = text.lower().strip()
    return [p for p in patterns if re.search(p, text_lower)]


def classify_intent(question: str) -> RouteResult:
    """Classify user question intent.

    Args:
        question: User's question text

    Returns:
        RouteResult with intent, suggested tools, confidence and reason
    """
    question_lower = question.lower().strip()

    # 1. Check chitchat first (highest priority for simple greetings)
    if _match_patterns(question, CHITCHAT_PATTERNS):
        return RouteResult(
            intent=Intent.CHITCHAT,
            suggested_tools=[],
            confidence=0.95,
            reason="Приветствие или благодарность",
        )

    # 2. Check personal progress (high confidence, no RAG needed)
    if _match_patterns(question, PERSONAL_PROGRESS_PATTERNS):
        # Determine which personal tool to use
        tools = []
        if re.search(r"(прогресс|статус|этап|чек-?лист|где я|на каком|что.*(сделал|прош))", question_lower):
            tools.append("get_my_progress")
        if re.search(r"(дальше|следующ|делать|начать|на каком)", question_lower):
            tools.append("get_next_steps")
        if re.search(r"(профиль|знаешь|помнишь|данные|информаци)", question_lower):
            tools.append("get_my_profile")
        if re.search(r"(мо(й|и|я).*(универ|город|программ)|сохранён)", question_lower):
            tools.append("get_my_entities")

        if not tools:
            tools = ["get_my_progress", "get_next_steps"]

        return RouteResult(
            intent=Intent.TOOL_ONLY,
            suggested_tools=tools,
            confidence=0.9,
            reason="Вопрос о персональном прогрессе",
        )

    # 3. Check calculator patterns
    if _match_patterns(question, CALCULATOR_PATTERNS):
        tools = []
        if re.search(r"(дедлайн|срок|deadline|когда.*(подавать|подать))", question_lower):
            tools.append("check_deadline")
        if re.search(r"(бюджет|стоимость|расход|сколько.*(стоит|денег|нужно))", question_lower):
            tools.append("calculate_budget")
        if re.search(r"(сколько|успе).*(дней|времени)", question_lower):
            tools.append("calculate_days_until")

        if not tools:
            tools = ["check_deadline"]

        # If university mentioned, pure tool. Otherwise might need RAG too.
        has_specific_uni = re.search(
            r"(uni wien|tu wien|wu wien|boku|meduni|jku|tu graz|uni graz|uni innsbruck|uni salzburg)",
            question_lower
        )

        if has_specific_uni:
            return RouteResult(
                intent=Intent.TOOL_ONLY,
                suggested_tools=tools,
                confidence=0.85,
                reason="Конкретный расчёт с указанным университетом",
            )
        else:
            return RouteResult(
                intent=Intent.TOOL_THEN_RAG,
                suggested_tools=tools,
                confidence=0.7,
                reason="Расчёт, возможно нужен контекст из базы знаний",
            )

    # 4. Check document patterns
    if _match_patterns(question, DOCUMENT_PATTERNS):
        # Check if asking about personal progress vs general info
        if re.search(r"(мо(й|и|я)|мне)", question_lower):
            return RouteResult(
                intent=Intent.TOOL_ONLY,
                suggested_tools=["get_my_progress", "get_document_checklist"],
                confidence=0.8,
                reason="Персональный чек-лист документов",
            )
        else:
            return RouteResult(
                intent=Intent.TOOL_THEN_RAG,
                suggested_tools=["get_document_checklist"],
                confidence=0.75,
                reason="Общий чек-лист + детали из базы знаний",
            )

    # 5. Check knowledge base patterns
    if _match_patterns(question, KNOWLEDGE_PATTERNS):
        return RouteResult(
            intent=Intent.RAG_ONLY,
            suggested_tools=[],
            confidence=0.85,
            reason="Нужна информация из базы знаний",
        )

    # 6. Default: RAG for safety (better to search than miss info)
    # But with lower confidence so the system knows it's a guess
    return RouteResult(
        intent=Intent.RAG_ONLY,
        suggested_tools=[],
        confidence=0.5,
        reason="Не удалось точно классифицировать, используем базу знаний",
    )


def should_use_rag(route: RouteResult) -> bool:
    """Check if RAG should be used for this route."""
    return route.intent in (Intent.RAG_ONLY, Intent.TOOL_THEN_RAG)


def should_use_tools(route: RouteResult) -> bool:
    """Check if tools should be used for this route."""
    return route.intent in (Intent.TOOL_ONLY, Intent.TOOL_THEN_RAG)


def is_chitchat(route: RouteResult) -> bool:
    """Check if this is a chitchat message."""
    return route.intent == Intent.CHITCHAT


# ── Quick Response Templates for Chitchat ──────────────────────────────


CHITCHAT_RESPONSES = {
    "greeting": [
        "Привет! Я помогу тебе с поступлением в австрийский вуз. Спрашивай!",
        "Привет! Чем могу помочь сегодня?",
        "Здравствуй! Готов помочь с любыми вопросами о поступлении.",
    ],
    "thanks": [
        "Пожалуйста! Если будут ещё вопросы — обращайся.",
        "Рад помочь! Удачи с поступлением!",
        "Не за что! Спрашивай, если что-то ещё понадобится.",
    ],
    "bye": [
        "Пока! Удачи с поступлением!",
        "До встречи! Возвращайся, если будут вопросы.",
        "Успехов! Буду ждать новостей о твоём поступлении.",
    ],
    "ok": [
        "Отлично! Если будут вопросы — я тут.",
        "Хорошо! Спрашивай, если что.",
    ],
}


def get_chitchat_response(question: str) -> str:
    """Get a quick response for chitchat messages."""
    import random

    q_lower = question.lower().strip()

    if re.search(r"(привет|здравствуй|добр|хай|хей|hello|hi|hey)", q_lower):
        return random.choice(CHITCHAT_RESPONSES["greeting"])
    elif re.search(r"(спасибо|благодар|thanks|thank|данке|danke)", q_lower):
        return random.choice(CHITCHAT_RESPONSES["thanks"])
    elif re.search(r"(пока|до свидания|bye|чао|ciao)", q_lower):
        return random.choice(CHITCHAT_RESPONSES["bye"])
    else:
        return random.choice(CHITCHAT_RESPONSES["ok"])
