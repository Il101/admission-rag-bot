from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph

from crag.retrievers.base import PipelineRetrieverBase


class PipelineBase(ABC):

    @property
    @abstractmethod
    def pipe_retriever(self) -> PipelineRetrieverBase:
        pass

    @property
    @abstractmethod
    def llm(self) -> BaseLanguageModel:
        pass

    @cached_property
    def graph(self) -> Runnable:
        graph = self.construct_graph()
        compiled_graph = graph.compile()
        return compiled_graph

    @abstractmethod
    def construct_graph(self) -> StateGraph:
        pass


class SimpleRagGraphState(TypedDict):
    question: str
    chat_history: str
    memory_context: str
    current_date: str
    user_data: dict
    do_generate: bool
    failed: bool
    remaining_rewrites: int
    documents: List[Document]
    generation: str


async def giveup(state: SimpleRagGraphState) -> SimpleRagGraphState:
    """Context-aware giveup: acknowledges user's question and suggests alternatives."""
    question = state.get("question", "")
    memory_context = state.get("memory_context", "")

    # Build a helpful fallback instead of a static message
    response = (
        "К сожалению, я не смог найти релевантную информацию по этому запросу "
        "в моей базе знаний. 😔\n\n"
    )

    # Suggest alternatives based on what hasn't been discussed
    if memory_context and "Ещё не обсуждали" in memory_context:
        response += (
            "Но я могу помочь с другими вопросами о поступлении! "
            "Попробуй спросить что-то из предложенных тем ниже. 👇"
        )
    else:
        response += (
            "Попробуй переформулировать вопрос или спросить что-то "
            "из предложенных тем ниже — я постараюсь помочь! 👇"
        )

    state["generation"] = response
    state["failed"] = True
    return state


def documents_to_context_str(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)
