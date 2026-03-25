"""
Base Agent class for specialized domain agents.

Provides common functionality for all specialized agents,
including context building and response generation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, AsyncGenerator

from .router import AgentType

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for specialized domain agents.

    Each specialized agent inherits from this and can override
    methods to customize behavior for its domain.
    """

    agent_type: AgentType = AgentType.GENERAL

    def __init__(self, rag: Any):
        """Initialize the agent with a RAG instance.

        Args:
            rag: SimpleRAG instance for retrieval and generation
        """
        self.rag = rag
        self._specialized_prompt = self._get_specialized_prompt()

    def _get_specialized_prompt(self) -> str:
        """Get domain-specific prompt additions.

        Override in subclasses for custom prompts.
        """
        from .router import get_router
        router = get_router()
        return router.get_specialized_prompt_prefix(self.agent_type)

    def enhance_system_prompt(self, base_prompt: str) -> str:
        """Enhance the base system prompt with domain-specific instructions.

        Args:
            base_prompt: The base RAG system prompt

        Returns:
            Enhanced prompt with specialized instructions
        """
        if not self._specialized_prompt:
            return base_prompt

        return f"""{self._specialized_prompt}

---

{base_prompt}"""

    def get_retrieval_boost_keywords(self) -> List[str]:
        """Get keywords to boost in retrieval for this domain.

        Override in subclasses to improve retrieval precision.
        """
        return []

    async def preprocess_question(self, question: str) -> str:
        """Preprocess the question before retrieval.

        Override in subclasses for domain-specific preprocessing.
        """
        return question

    async def postprocess_answer(self, answer: str) -> str:
        """Postprocess the generated answer.

        Override in subclasses for domain-specific formatting.
        """
        return answer

    async def answer(
        self,
        question: str,
        context_str: str,
        chat_history: str,
        memory_context: str,
        current_date: str,
    ) -> AsyncGenerator[str, None]:
        """Generate an answer using the specialized agent.

        By default, delegates to the RAG's streaming method with
        an enhanced system prompt.
        """
        # Preprocess question
        processed_question = await self.preprocess_question(question)

        # Stream answer
        async for chunk in self.rag.astream_answer(
            processed_question,
            context_str,
            chat_history,
            memory_context,
            current_date,
        ):
            yield chunk


class DocumentAgent(BaseAgent):
    """Specialized agent for document-related questions."""
    agent_type = AgentType.DOCUMENT

    def get_retrieval_boost_keywords(self) -> List[str]:
        return [
            "апостиль", "нострификация", "перевод", "документ",
            "attestat", "diploma", "translation",
        ]


class UniversityAgent(BaseAgent):
    """Specialized agent for university-related questions."""
    agent_type = AgentType.UNIVERSITY

    def get_retrieval_boost_keywords(self) -> List[str]:
        return [
            "университет", "программа", "дедлайн", "подача",
            "zulassung", "bewerbung", "application",
        ]


class VisaAgent(BaseAgent):
    """Specialized agent for visa and residence permit questions."""
    agent_type = AgentType.VISA

    def get_retrieval_boost_keywords(self) -> List[str]:
        return [
            "виза", "ВНЖ", "aufenthaltstitel", "MA35",
            "residence permit", "страховка",
        ]


class FinanceAgent(BaseAgent):
    """Specialized agent for financial questions."""
    agent_type = AgentType.FINANCE

    def get_retrieval_boost_keywords(self) -> List[str]:
        return [
            "стипендия", "бюджет", "стоимость", "oead",
            "scholarship", "sperrkonto", "блокированный счёт",
        ]


class HousingAgent(BaseAgent):
    """Specialized agent for housing questions."""
    agent_type = AgentType.HOUSING

    def get_retrieval_boost_keywords(self) -> List[str]:
        return [
            "жильё", "общежитие", "studentenheim", "квартира",
            "wohnung", "аренда", "meldezettel",
        ]


def create_agent(agent_type: AgentType, rag: Any) -> BaseAgent:
    """Factory function to create the appropriate agent.

    Args:
        agent_type: Type of agent to create
        rag: SimpleRAG instance

    Returns:
        Specialized agent instance
    """
    agent_classes = {
        AgentType.DOCUMENT: DocumentAgent,
        AgentType.UNIVERSITY: UniversityAgent,
        AgentType.VISA: VisaAgent,
        AgentType.FINANCE: FinanceAgent,
        AgentType.HOUSING: HousingAgent,
        AgentType.GENERAL: BaseAgent,
    }

    agent_class = agent_classes.get(agent_type, BaseAgent)

    # BaseAgent is abstract, so for GENERAL we need a concrete implementation
    if agent_type == AgentType.GENERAL:
        class GeneralAgent(BaseAgent):
            agent_type = AgentType.GENERAL
        return GeneralAgent(rag)

    return agent_class(rag)
