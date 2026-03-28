"""
Pipeline architecture for the RAG system.

Breaks down the monolithic handler into composable steps for better
maintainability, testability, and extensibility.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# ── Pipeline Context ─────────────────────────────────────────────────────


@dataclass
class PipelineContext:
    """Shared state across pipeline steps.

    Holds all data flowing through the pipeline, from input to output.
    Each step reads from and writes to this context.
    """
    # Input
    question: str
    tg_id: int
    onboarding_data: dict = field(default_factory=dict)

    # Memory layers
    memory_context: str = ""
    chat_history: str = ""
    journey_state: dict = field(default_factory=dict)
    conversation_summary: str = ""

    # Processing state
    rewritten_question: str = ""
    embedding: List[float] = field(default_factory=list)
    retrieved_docs: List[Any] = field(default_factory=list)
    graded_docs: List[Any] = field(default_factory=list)
    reranked_docs: List[Any] = field(default_factory=list)
    context_str: str = ""

    # Tools
    tool_needed: Optional[dict] = None
    tool_result: Optional[dict] = None

    # Output
    answer_json: str = ""
    answer_text: str = ""
    suggested_questions: List[str] = field(default_factory=list)

    # Metadata
    cache_hit: bool = False
    timings: dict = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # Pipeline control
    should_stop: bool = False  # Set to True to short-circuit pipeline


# ── Pipeline Step Base ───────────────────────────────────────────────────


class PipelineStep(ABC):
    """Base class for pipeline steps.

    Each step performs one logical operation in the RAG pipeline.
    Steps can read/write to the context and signal early termination.
    """
    name: str = "base_step"

    @abstractmethod
    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        """Execute the step, modifying ctx in place.

        Args:
            ctx: Pipeline context to read from and write to
            rag: SimpleRAG instance for LLM/DB operations

        Set ctx.should_stop = True to terminate pipeline early.
        """
        pass


# ── Concrete Steps ───────────────────────────────────────────────────────


class RewriteStep(PipelineStep):
    """Rewrite query to resolve anaphora and add keywords."""
    name = "rewrite"

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        t0 = time.monotonic()
        ctx.rewritten_question = await rag.arewrite_query(
            ctx.question, ctx.chat_history, ctx.memory_context
        )
        ctx.timings["rewrite"] = time.monotonic() - t0


class CacheStep(PipelineStep):
    """Check semantic answer cache."""
    name = "cache"

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        # If a tool call is required, skip semantic cache to avoid
        # returning a stale answer that doesn't include tool output.
        if ctx.tool_needed:
            return

        t0 = time.monotonic()
        cached = await rag.get_cached_answer(ctx.rewritten_question)
        ctx.timings["cache_check"] = time.monotonic() - t0

        if cached:
            ctx.answer_json = cached
            ctx.cache_hit = True
            ctx.should_stop = True  # Early exit
            logger.info(f"[User {ctx.tg_id}] Cache hit")


class ToolCheckStep(PipelineStep):
    """Check if a tool/function call is needed."""
    name = "tool_check"

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        t0 = time.monotonic()
        ctx.tool_needed = await rag.acheck_tool_need(
            ctx.rewritten_question, ctx.memory_context
        )
        ctx.timings["tool_check"] = time.monotonic() - t0


class ToolExecuteStep(PipelineStep):
    """Execute tool if needed."""
    name = "tool_execute"

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        if not ctx.tool_needed:
            return

        t0 = time.monotonic()
        ctx.tool_result = await rag.aexecute_tool(
            ctx.tool_needed["tool_name"],
            ctx.tool_needed["arguments"],
        )
        ctx.timings["tool_execute"] = time.monotonic() - t0
        logger.info(
            f"[User {ctx.tg_id}] Tool executed: {ctx.tool_needed['tool_name']}"
        )


class RetrieveStep(PipelineStep):
    """Retrieve relevant documents."""
    name = "retrieve"

    def __init__(
        self,
        use_hyde: bool = True,
        use_decomposition: bool = False,
        use_smart_retrieval: bool = True,
        top_k: int = 6,
    ):
        self.use_hyde = use_hyde
        self.use_decomposition = use_decomposition
        self.use_smart_retrieval = use_smart_retrieval
        self.top_k = top_k

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        t0 = time.monotonic()

        user_filters = self._build_user_filters(ctx.onboarding_data)

        if self.use_decomposition:
            ctx.retrieved_docs = await rag.aretrieve_decomposed(
                ctx.rewritten_question,
                top_k=self.top_k,
                user_filters=user_filters,
                use_hyde=self.use_hyde,
                memory_context=ctx.memory_context,
            )
        elif self.use_smart_retrieval:
            ctx.retrieved_docs = await rag.aretrieve_smart(
                ctx.rewritten_question,
                top_k=self.top_k,
                user_filters=user_filters,
                use_hyde=self.use_hyde,
                memory_context=ctx.memory_context,
            )
        else:
            ctx.retrieved_docs = await rag.aretrieve(
                ctx.rewritten_question,
                top_k=self.top_k,
                user_filters=user_filters,
                use_hyde=self.use_hyde,
                memory_context=ctx.memory_context,
            )

        ctx.timings["retrieve"] = time.monotonic() - t0
        logger.info(
            f"[User {ctx.tg_id}] Retrieved {len(ctx.retrieved_docs)} docs"
        )

    @staticmethod
    def _build_user_filters(onboarding_data: dict) -> dict:
        """Build metadata filters from user's onboarding profile."""
        filters = {}

        country = onboarding_data.get("countryScope") or onboarding_data.get("country")
        if country:
            # Map common country names to codes
            country_map = {
                "россия": "RU", "russia": "RU", "ru": "RU",
                "украина": "UA", "ukraine": "UA", "ua": "UA",
                "беларусь": "BY", "belarus": "BY", "by": "BY",
                "казахстан": "KZ", "kazakhstan": "KZ", "kz": "KZ",
            }
            code = country_map.get(country.lower().strip(), country.upper())
            if len(code) == 2:
                filters["country_scope"] = code

        target_level = onboarding_data.get("targetLevel")
        if target_level:
            level_map = {"bachelor": "bachelor", "master": "master", "phd": "phd"}
            code = level_map.get(target_level.lower().strip(), "")
            if code:
                filters["level"] = code

        return filters


class GradeStep(PipelineStep):
    """Grade documents for relevance (CRAG filtering)."""
    name = "grade"

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        if not ctx.retrieved_docs:
            ctx.should_stop = True
            ctx.answer_json = json.dumps({
                "answer": "К сожалению, я не смог найти релевантную информацию по этому запросу. 😔",
                "suggested_questions": [],
            }, ensure_ascii=False)
            return

        t0 = time.monotonic()
        ctx.graded_docs = await rag.agrade_documents(
            ctx.rewritten_question, ctx.retrieved_docs
        )
        ctx.timings["grade"] = time.monotonic() - t0

        # Collect sources
        ctx.sources = [
            doc.metadata.get("source_url", "")
            for doc in ctx.graded_docs
            if doc.metadata.get("source_url")
        ]

        if not ctx.graded_docs:
            ctx.should_stop = True
            ctx.answer_json = json.dumps({
                "answer": "К сожалению, я не смог найти релевантную информацию по этому запросу. 😔",
                "suggested_questions": [],
            }, ensure_ascii=False)
            return

        logger.info(
            f"[User {ctx.tg_id}] Graded: {len(ctx.retrieved_docs)} → {len(ctx.graded_docs)} docs"
        )


class RerankStep(PipelineStep):
    """Re-rank documents using LLM scoring."""
    name = "rerank"

    def __init__(self, top_k: int = 6):
        self.top_k = top_k

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        if not ctx.graded_docs or len(ctx.graded_docs) <= self.top_k:
            ctx.reranked_docs = ctx.graded_docs
            return

        t0 = time.monotonic()
        ctx.reranked_docs = await rag.arerank_documents(
            ctx.rewritten_question, ctx.graded_docs, top_k=self.top_k
        )
        ctx.timings["rerank"] = time.monotonic() - t0


class BuildContextStep(PipelineStep):
    """Build context string from documents."""
    name = "build_context"

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        from crag.simple_rag import documents_to_context_str

        docs = ctx.reranked_docs or ctx.graded_docs or ctx.retrieved_docs
        ctx.context_str = documents_to_context_str(docs)


class GenerateStep(PipelineStep):
    """Generate answer using LLM."""
    name = "generate"

    def __init__(self, stream_callback: Optional[Callable[[str], Awaitable[None]]] = None):
        self.stream_callback = stream_callback

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        from bot.memory import get_current_date

        t0 = time.monotonic()
        accumulated = ""

        # Choose streaming method based on tool result
        if ctx.tool_result:
            stream_gen = rag.astream_answer_with_tools(
                ctx.rewritten_question,
                ctx.context_str,
                ctx.chat_history,
                ctx.memory_context,
                get_current_date(),
                tool_result=ctx.tool_result,
            )
        else:
            stream_gen = rag.astream_answer(
                ctx.rewritten_question,
                ctx.context_str,
                ctx.chat_history,
                ctx.memory_context,
                get_current_date(),
            )

        async for chunk in stream_gen:
            accumulated += chunk
            if self.stream_callback:
                try:
                    await self.stream_callback(accumulated)
                except Exception as e:
                    logger.warning(f"Stream callback error: {e}")

        ctx.answer_json = accumulated
        ctx.timings["generate"] = time.monotonic() - t0


class CacheStoreStep(PipelineStep):
    """Store answer in semantic cache."""
    name = "cache_store"

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        if not ctx.cache_hit and ctx.answer_json:
            # Fire and forget
            asyncio.create_task(
                rag.cache_answer(ctx.rewritten_question, ctx.answer_json)
            )


# ── Pipeline Orchestrator ────────────────────────────────────────────────


class Pipeline:
    """Orchestrates execution of pipeline steps."""

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    async def run(self, ctx: PipelineContext, rag: Any) -> PipelineContext:
        """Execute all steps in sequence.

        Stops early if ctx.should_stop is set by any step.

        Args:
            ctx: Pipeline context
            rag: SimpleRAG instance

        Returns:
            Modified context after pipeline execution
        """
        start = time.monotonic()

        for step in self.steps:
            if ctx.should_stop:
                logger.debug(f"Pipeline stopped before {step.name}")
                break

            try:
                await step.execute(ctx, rag)
            except Exception as e:
                ctx.error = f"{type(e).__name__}: {e}"
                logger.exception(f"Pipeline step '{step.name}' failed")
                raise

        ctx.timings["total"] = time.monotonic() - start
        return ctx


# ── Pipeline Factories ───────────────────────────────────────────────────


def create_default_pipeline(
    stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    use_hyde: bool = True,
    use_decomposition: bool = False,
    use_smart_retrieval: bool = True,
    use_reranking: bool = True,
    use_tools: bool = True,
    top_k: int = 10,
    rerank_top_k: int = 6,
) -> Pipeline:
    """Create the default RAG pipeline.

    Args:
        stream_callback: Async callback for streaming output
        use_hyde: Enable HyDE for retrieval
        use_decomposition: Enable query decomposition
        use_smart_retrieval: Enable FACT/NARRATIVE smart retrieval routing
        use_reranking: Enable LLM re-ranking
        use_tools: Enable tool/function calling
        top_k: Number of documents to retrieve before grading/reranking
        rerank_top_k: Number of documents to keep after re-ranking

    Returns:
        Configured Pipeline instance
    """
    if use_tools:
        steps = [
            RewriteStep(),
            ToolCheckStep(),
            ToolExecuteStep(),
            CacheStep(),
        ]
    else:
        steps = [
            RewriteStep(),
            CacheStep(),
        ]

    steps.extend([
        RetrieveStep(
            use_hyde=use_hyde,
            use_decomposition=use_decomposition,
            use_smart_retrieval=use_smart_retrieval,
            top_k=top_k,
        ),
        GradeStep(),
    ])

    if use_reranking:
        steps.append(RerankStep(top_k=rerank_top_k))

    steps.extend([
        BuildContextStep(),
        GenerateStep(stream_callback=stream_callback),
        CacheStoreStep(),
    ])

    return Pipeline(steps)


def create_fast_pipeline(
    stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Pipeline:
    """Create a fast pipeline without advanced features.

    Disables HyDE, decomposition, reranking, and tools for speed.
    Useful for simple questions or when latency is critical.
    """
    return Pipeline([
        RewriteStep(),
        CacheStep(),
        RetrieveStep(use_hyde=False, use_decomposition=False, top_k=4),
        GradeStep(),
        BuildContextStep(),
        GenerateStep(stream_callback=stream_callback),
        CacheStoreStep(),
    ])


def create_research_pipeline(
    stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Pipeline:
    """Create a thorough pipeline for complex research questions.

    Enables all features with higher top_k for comprehensive coverage.
    """
    return Pipeline([
        RewriteStep(),
        CacheStep(),
        ToolCheckStep(),
        ToolExecuteStep(),
        RetrieveStep(use_hyde=True, use_decomposition=True, top_k=10),
        GradeStep(),
        RerankStep(top_k=8),
        BuildContextStep(),
        GenerateStep(stream_callback=stream_callback),
        CacheStoreStep(),
    ])
