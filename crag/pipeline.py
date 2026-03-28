"""
Pipeline architecture for the RAG system.

Breaks down the monolithic handler into composable steps for better
maintainability, testability, and extensibility.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Callable, Awaitable

logger = logging.getLogger(__name__)


_HOUSING_KEYWORDS = (
    "жиль", "общежит", "studentenheim", "wg ", "квартир", "аренд",
    "wohnung", "miet", "meldezettel", "переезд",
)
_FOOD_KEYWORDS = (
    "питан", "еда", "столов", "mensa", "кафе", "ресторан", "перекус",
    "vegan", "vegetarian", "веган", "вегетари", "canteen", "cafeteria",
    "meal", "меню",
)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def _has_topic(doc: Any, topic: str) -> bool:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("topic", "")).strip().lower() == topic


def _doc_is_housing(doc: Any) -> bool:
    metadata = getattr(doc, "metadata", {}) or {}
    content = getattr(doc, "page_content", "") or ""
    text = f"{content}\n{json.dumps(metadata, ensure_ascii=False)}"
    return _has_topic(doc, "housing") or _contains_any(text, _HOUSING_KEYWORDS)


def _doc_is_food(doc: Any) -> bool:
    metadata = getattr(doc, "metadata", {}) or {}
    content = getattr(doc, "page_content", "") or ""
    text = f"{content}\n{json.dumps(metadata, ensure_ascii=False)}"
    return _contains_any(text, _FOOD_KEYWORDS)


def apply_entity_grounding_guardrails(question: str, docs: List[Any]) -> List[Any]:
    """Prevent cross-entity fact mixing in final context.

    For housing questions, removes food-only chunks (e.g., Mensa/menus)
    to avoid leaking unrelated claims into housing answers.
    """
    if not docs:
        return docs

    q = (question or "").lower()
    asks_housing = _contains_any(q, _HOUSING_KEYWORDS)

    # Main safety net for the observed failure mode:
    # housing question + accidental food chunk contamination.
    # Applied even if question contains food words to prevent "Mensa -> dormitory"
    # leakage unless the chunk itself is explicitly housing-related.
    if asks_housing:
        filtered = [doc for doc in docs if not (_doc_is_food(doc) and not _doc_is_housing(doc))]
        if filtered:
            removed = len(docs) - len(filtered)
            if removed > 0:
                logger.info(
                    "Grounding guardrail removed %d food-only docs for housing question",
                    removed,
                )
            return filtered

    return docs


def _get_rerank_soft_timeout_sec() -> float:
    """Soft timeout for reranking; 0 disables timeout."""
    raw = os.getenv("RERANK_SOFT_TIMEOUT_SEC", "0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, value)


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


class TagBoostStep(PipelineStep):
    """Apply tag-based boosting to documents.

    Uses domain tags to boost relevance of documents that match query tags.
    This helps prioritize documents from the correct domain (housing vs food, etc.).
    """
    name = "tag_boost"

    def __init__(self, boost_factor: float = 0.3):
        """Initialize tag boost step.

        Args:
            boost_factor: Weight of tag boost (0.0 = no boost, 1.0 = full boost)
        """
        self.boost_factor = boost_factor

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        if not ctx.graded_docs:
            return

        from crag.tag_set_layer import tag_based_reranking

        t0 = time.monotonic()

        # Convert docs to dict format for tag_based_reranking
        docs_as_dicts = []
        for doc in ctx.graded_docs:
            docs_as_dicts.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 0.5,  # Use neutral score, we care about relative reranking
            })

        # Apply tag-based reranking
        ranked = tag_based_reranking(
            docs_as_dicts,
            ctx.rewritten_question or ctx.question,
            factor=self.boost_factor,
        )

        # Sort graded_docs based on tag-adjusted scores
        doc_order = {id(doc_dict["content"]): i for i, (doc_dict, _) in enumerate(ranked)}

        # Reorder graded_docs
        ctx.graded_docs = sorted(
            ctx.graded_docs,
            key=lambda doc: doc_order.get(id(doc.page_content), 999),
        )

        ctx.timings["tag_boost"] = time.monotonic() - t0

        logger.info(
            f"[User {ctx.tg_id}] Tag boost applied with factor={self.boost_factor}"
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
        timeout = _get_rerank_soft_timeout_sec()
        if timeout > 0:
            try:
                ctx.reranked_docs = await asyncio.wait_for(
                    rag.arerank_documents(
                        ctx.rewritten_question, ctx.graded_docs, top_k=self.top_k
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.info(
                    f"[User {ctx.tg_id}] Rerank soft-timeout after {timeout:.2f}s, keeping graded order"
                )
                ctx.reranked_docs = ctx.graded_docs[: self.top_k]
        else:
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
        docs = apply_entity_grounding_guardrails(ctx.rewritten_question, docs)
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


class AssertionCheckStep(PipelineStep):
    """Validate answer assertions against source categories.

    Prevents cross-entity hallucinations by checking that answer claims
    are grounded in sources from appropriate categories.
    """
    name = "assertion_check"

    async def execute(self, ctx: PipelineContext, rag: Any) -> None:
        from crag.assertion_validator import validate_answer_assertions, add_assertion_disclaimer

        # Only validate if we have both answer and sources
        if not ctx.answer_json or not ctx.reranked_docs:
            return

        # Extract answer text from JSON (may contain {"answer": "...", "suggested_questions": [...]})
        try:
            answer_data = json.loads(ctx.answer_json)
            answer_text = answer_data.get("answer", ctx.answer_json)
        except json.JSONDecodeError:
            answer_text = ctx.answer_json

        # Skip if answer is too short (likely error message)
        if len(answer_text) < 50:
            return

        t0 = time.monotonic()
        result = await validate_answer_assertions(
            answer_text,
            ctx.reranked_docs,
            ctx.rewritten_question or ctx.question,
            llm_provider=None,  # Can pass rag.llm for advanced validation
        )
        ctx.timings["assertion_check"] = time.monotonic() - t0

        # If validation failed, add disclaimer
        if not result.is_valid and result.warnings:
            logger.warning(
                f"[User {ctx.tg_id}] Assertion check failed: {'; '.join(result.warnings)}"
            )

            # Add disclaimer to answer
            answer_with_disclaimer = add_assertion_disclaimer(answer_text, result.warnings)

            # Update answer in JSON
            if answer_data and isinstance(answer_data, dict):
                answer_data["answer"] = answer_with_disclaimer
                ctx.answer_json = json.dumps(answer_data, ensure_ascii=False)
            else:
                ctx.answer_json = answer_with_disclaimer


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
        TagBoostStep(boost_factor=0.3),  # Apply tag-based boosting
    ])

    if use_reranking:
        steps.append(RerankStep(top_k=rerank_top_k))

    steps.extend([
        BuildContextStep(),
        GenerateStep(stream_callback=stream_callback),
        AssertionCheckStep(),
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
        TagBoostStep(boost_factor=0.2),  # Lighter boost for fast pipeline
        BuildContextStep(),
        GenerateStep(stream_callback=stream_callback),
        AssertionCheckStep(),
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
        TagBoostStep(boost_factor=0.4),  # Stronger boost for research pipeline
        RerankStep(top_k=8),
        BuildContextStep(),
        GenerateStep(stream_callback=stream_callback),
        AssertionCheckStep(),
        CacheStoreStep(),
    ])
