"""
Multi-Agent Architecture for the admission bot.

Provides specialized agents for different domains (documents, universities, visa)
with a router agent that dispatches questions to the appropriate specialist.
"""

from .router import RouterAgent, AgentType
from .base import BaseAgent

__all__ = ["RouterAgent", "AgentType", "BaseAgent"]
