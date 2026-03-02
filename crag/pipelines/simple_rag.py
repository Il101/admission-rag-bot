from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from crag.pipelines.base import (
    PipelineBase,
    SimpleRagGraphState,
    documents_to_context_str,
)
from crag.retrievers.base import PipelineRetrieverBase


class SimpleRAG(PipelineBase):

    def __init__(
        self,
        retriever: PipelineRetrieverBase,
        llm: BaseLanguageModel,
        rag_prompt: PromptTemplate,
    ) -> None:
        super().__init__()
        self._pipe_retriever = retriever
        self._llm = llm
        self._rag_chain = rag_prompt | llm | StrOutputParser()

    @property
    def pipe_retriever(self) -> PipelineRetrieverBase:
        return self._pipe_retriever

    @property
    def llm(self) -> BaseLanguageModel:
        return self._llm

    @property
    def rag_chain(self):
        """Expose the RAG chain for streaming generation outside the graph."""
        return self._rag_chain

    async def retrieve(self, state: SimpleRagGraphState) -> SimpleRagGraphState:
        question = state["question"]

        documents = await self._pipe_retriever.retriever.ainvoke(question)

        state["documents"] = documents
        return state

    async def generate(self, state: SimpleRagGraphState) -> SimpleRagGraphState:
        question = state["question"]
        documents = state["documents"]
        chat_history = state.get("chat_history", "История пуста.")
        memory_context = state.get("memory_context", "Новый пользователь.")
        current_date = state.get("current_date", "")

        context = documents_to_context_str(documents)
        generation = await self._rag_chain.ainvoke(
            {
                "context": context, 
                "question": question, 
                "chat_history": chat_history,
                "memory_context": memory_context,
                "current_date": current_date,
            }
        )

        state["generation"] = generation
        return state

    async def decide_to_generate(self, state: SimpleRagGraphState) -> str:
        if state["do_generate"]:
            return "generate"
        else:
            return "stop"

    def construct_graph(self) -> StateGraph:
        workflow = StateGraph(SimpleRagGraphState)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)

        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self.decide_to_generate,
            {
                "stop": END,
                "generate": "generate",
            },
        )
        workflow.add_edge("generate", END)

        return workflow
