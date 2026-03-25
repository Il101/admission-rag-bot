"""
Router Agent for dispatching questions to specialized agents.

Analyzes the user's question and routes it to the most appropriate
domain-specific agent based on topic detection.
"""

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialized agents."""
    DOCUMENT = "document"       # Documents, apostille, translations, nostrification
    UNIVERSITY = "university"   # Universities, programs, deadlines, applications
    VISA = "visa"               # Visa, residence permit (ВНЖ), MA35, insurance
    FINANCE = "finance"         # Tuition, scholarships, budget, blocked account
    HOUSING = "housing"         # Housing, relocation, registration
    GENERAL = "general"         # General questions, unclear topics


# Routing keywords for each agent type
ROUTING_KEYWORDS: dict[AgentType, list[str]] = {
    AgentType.DOCUMENT: [
        "документ", "аттестат", "диплом", "апостиль", "перевод", "нотариальн",
        "легализ", "копи", "оригинал", "выписк", "приложени", "нострификац",
        "признание", "подтвержден", "эквивалент", "anerkenn", "nostrifik",
        "справк", "сертификат", "transcript",
    ],
    AgentType.UNIVERSITY: [
        "университет", "вуз", "uni ", "fachhochschule", "программ", "факультет",
        "специальност", "tu wien", "uni wien", "uni graz", "uni innsbruck",
        "boku", "wu wien", "meduni", "jku", "выбор", "рейтинг", "подача",
        "заявк", "application", "zulassung", "bewerbung", "дедлайн", "срок",
        "зачислен", "admission", "offer", "приём", "принят",
    ],
    AgentType.VISA: [
        "виз", "внж", "вид на жительство", "aufenthalts", "residence", "permit",
        "ma35", "магистрат", "посольств", "консульств", "страховк", "гарант",
        "d визa", "студенческ", "разрешени",
    ],
    AgentType.FINANCE: [
        "стоимост", "стипенди", "оплат", "studiengebühr", "взнос", "scholarship",
        "oead", "грант", "бюджет", "расход", "сколько стоит", "финанс", "банк",
        "блокированн", "sperrkonto", "студенческий счёт", "деньг",
    ],
    AgentType.HOUSING: [
        "жильё", "жилье", "квартир", "общежити", "studentenheim", "wg ",
        "wohnung", "miet", "переезд", "аренд", "регистрац", "мельдунг",
        "meldezettel", "снять", "поиск жиль",
    ],
}


class RouterAgent:
    """Routes questions to specialized agents based on topic detection."""

    def __init__(self):
        self._keyword_cache: dict[str, AgentType] = {}

    def route(self, question: str, memory_context: str = "") -> AgentType:
        """Determine which agent should handle the question.

        Uses keyword-based routing for speed. Falls back to GENERAL
        if no strong signal is detected.

        Args:
            question: The user's question
            memory_context: User context (can influence routing)

        Returns:
            AgentType indicating which specialist should handle this
        """
        combined_text = f"{question} {memory_context}".lower()

        # Quick cache check
        cache_key = question[:100].lower()
        if cache_key in self._keyword_cache:
            return self._keyword_cache[cache_key]

        # Score each agent type
        scores: dict[AgentType, int] = {agent: 0 for agent in AgentType}

        for agent_type, keywords in ROUTING_KEYWORDS.items():
            for keyword in keywords:
                if keyword in combined_text:
                    scores[agent_type] += 1

        # Find the best match
        best_agent = max(scores, key=scores.get)
        best_score = scores[best_agent]

        # If no strong signal, use GENERAL
        if best_score == 0:
            result = AgentType.GENERAL
        else:
            result = best_agent

        # Cache the result
        self._keyword_cache[cache_key] = result

        logger.info(
            f"Router: '{question[:50]}...' → {result.value} (score: {best_score})"
        )
        return result

    def get_agent_description(self, agent_type: AgentType) -> str:
        """Get a human-readable description of an agent's specialty."""
        descriptions = {
            AgentType.DOCUMENT: "Специалист по документам, апостилям, переводам и нострификации",
            AgentType.UNIVERSITY: "Специалист по вузам, программам, подаче документов и дедлайнам",
            AgentType.VISA: "Специалист по визам, ВНЖ и работе с MA35",
            AgentType.FINANCE: "Специалист по финансам, стипендиям и бюджету",
            AgentType.HOUSING: "Специалист по жилью и переезду",
            AgentType.GENERAL: "Общий консультант по поступлению",
        }
        return descriptions.get(agent_type, "Консультант")

    def get_specialized_prompt_prefix(self, agent_type: AgentType) -> str:
        """Get a prompt prefix to prime the LLM for the specialist role.

        This is injected into the system prompt to focus the agent's responses.
        """
        prefixes = {
            AgentType.DOCUMENT: """Ты — специалист по документам для поступления в Австрию.
Твоя экспертиза: апостили, нотариальные переводы, нострификация, легализация.
Отвечай с фокусом на правильное оформление документов, сроки и порядок действий.""",

            AgentType.UNIVERSITY: """Ты — специалист по австрийским университетам.
Твоя экспертиза: TU Wien, Uni Wien, WU Wien, Uni Graz и другие вузы.
Отвечай с фокусом на программы, дедлайны подачи, требования к поступлению.""",

            AgentType.VISA: """Ты — специалист по визам и ВНЖ в Австрии.
Твоя экспертиза: студенческая виза D, Aufenthaltstitel Student, MA35.
Отвечай с фокусом на требования, документы и сроки получения.""",

            AgentType.FINANCE: """Ты — специалист по финансовым вопросам обучения в Австрии.
Твоя экспертиза: стоимость обучения, стипендии OeAD, блокированный счёт, бюджет.
Отвечай с конкретными цифрами и практическими советами по экономии.""",

            AgentType.HOUSING: """Ты — специалист по жилью и переезду в Австрию.
Твоя экспертиза: студенческие общежития, аренда, регистрация (Meldezettel).
Отвечай с практическими советами по поиску жилья и первым шагам в Австрии.""",

            AgentType.GENERAL: """Ты — универсальный консультант по поступлению в Австрию.
Помогаешь с любыми вопросами и направляешь к нужным ресурсам.""",
        }
        return prefixes.get(agent_type, "")


# Global router instance
_router: Optional[RouterAgent] = None


def get_router() -> RouterAgent:
    """Get or create the global router instance."""
    global _router
    if _router is None:
        _router = RouterAgent()
    return _router
