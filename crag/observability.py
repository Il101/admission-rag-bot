"""
Observability module for RAG pipeline using LangFuse.

Provides decorators and utilities for tracing LLM calls, retrieval operations,
and full pipeline execution for debugging, monitoring, and optimization.
"""

import os
import logging
import functools
from typing import Optional, Callable, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# LangFuse client (lazy initialized)
_langfuse_client = None
_langfuse_enabled = False


def _get_langfuse():
    """Get or initialize LangFuse client."""
    global _langfuse_client, _langfuse_enabled

    if _langfuse_client is not None:
        return _langfuse_client

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.info("LangFuse credentials not found, observability disabled")
        _langfuse_enabled = False
        return None

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        _langfuse_enabled = True
        logger.info("LangFuse observability enabled")
        return _langfuse_client
    except ImportError:
        logger.warning("langfuse package not installed, observability disabled")
        _langfuse_enabled = False
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize LangFuse: {e}")
        _langfuse_enabled = False
        return None


def is_observability_enabled() -> bool:
    """Check if observability is enabled."""
    _get_langfuse()  # Ensure initialized
    return _langfuse_enabled


class TraceContext:
    """Context manager for a single trace/span."""

    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self.name = name
        self.trace_id = trace_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self._trace = None
        self._span = None

    async def __aenter__(self):
        langfuse = _get_langfuse()
        if not langfuse:
            return self

        try:
            self._trace = langfuse.trace(
                name=self.name,
                id=self.trace_id,
                user_id=self.user_id,
                metadata=self.metadata,
            )
        except Exception as e:
            logger.debug(f"Failed to create trace: {e}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._trace:
            try:
                if exc_type:
                    self._trace.update(
                        output={"error": str(exc_val)},
                        level="ERROR",
                    )
                langfuse = _get_langfuse()
                if langfuse:
                    langfuse.flush()
            except Exception as e:
                logger.debug(f"Failed to finalize trace: {e}")

    def span(self, name: str, **kwargs):
        """Create a child span."""
        if not self._trace:
            return _DummySpan()
        try:
            return self._trace.span(name=name, **kwargs)
        except Exception:
            return _DummySpan()

    def generation(self, name: str, model: str, **kwargs):
        """Create a generation (LLM call) span."""
        if not self._trace:
            return _DummyGeneration()
        try:
            return self._trace.generation(name=name, model=model, **kwargs)
        except Exception:
            return _DummyGeneration()


class _DummySpan:
    """Dummy span when observability is disabled."""
    def end(self, **kwargs): pass
    def update(self, **kwargs): pass
    def event(self, **kwargs): pass


class _DummyGeneration:
    """Dummy generation when observability is disabled."""
    def end(self, **kwargs): pass
    def update(self, **kwargs): pass


def trace_llm_call(name: str, model: str = "gemini-2.5-flash"):
    """Decorator to trace LLM calls.

    Usage:
        @trace_llm_call("rewrite_query", "gemini-2.5-flash")
        async def arewrite_query(self, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            langfuse = _get_langfuse()
            if not langfuse:
                return await func(*args, **kwargs)

            # Extract input for logging
            input_data = {
                "args_preview": str(args[1:3])[:200] if len(args) > 1 else "",
                "kwargs_keys": list(kwargs.keys()),
            }

            generation = None
            try:
                generation = langfuse.generation(
                    name=name,
                    model=model,
                    input=input_data,
                )
            except Exception as e:
                logger.debug(f"Failed to create generation span: {e}")

            try:
                result = await func(*args, **kwargs)
                if generation:
                    try:
                        output_preview = str(result)[:500] if result else ""
                        generation.end(output=output_preview)
                    except Exception:
                        pass
                return result
            except Exception as e:
                if generation:
                    try:
                        generation.end(
                            output={"error": str(e)},
                            level="ERROR",
                        )
                    except Exception:
                        pass
                raise

        return wrapper
    return decorator


def trace_retrieval(name: str = "retrieval"):
    """Decorator to trace retrieval operations.

    Usage:
        @trace_retrieval("hybrid_search")
        async def aretrieve(self, query, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            langfuse = _get_langfuse()
            if not langfuse:
                return await func(*args, **kwargs)

            # Extract query from args/kwargs
            query = kwargs.get("query") or (args[1] if len(args) > 1 else "")
            top_k = kwargs.get("top_k", 6)

            span = None
            try:
                span = langfuse.span(
                    name=name,
                    input={
                        "query": str(query)[:200],
                        "top_k": top_k,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to create span: {e}")

            try:
                result = await func(*args, **kwargs)
                if span:
                    try:
                        docs_count = len(result) if hasattr(result, '__len__') else 0
                        span.end(
                            output={
                                "docs_count": docs_count,
                                "sources": [
                                    d.metadata.get("source_url", "")[:100]
                                    for d in (result or [])[:3]
                                ] if result else [],
                            }
                        )
                    except Exception:
                        pass
                return result
            except Exception as e:
                if span:
                    try:
                        span.end(output={"error": str(e)}, level="ERROR")
                    except Exception:
                        pass
                raise

        return wrapper
    return decorator


def trace_pipeline_step(step_name: str):
    """Decorator for tracing individual pipeline steps.

    Usage:
        @trace_pipeline_step("grade_documents")
        async def agrade_documents(self, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            langfuse = _get_langfuse()
            if not langfuse:
                return await func(*args, **kwargs)

            span = None
            try:
                span = langfuse.span(name=step_name)
            except Exception:
                pass

            try:
                result = await func(*args, **kwargs)
                if span:
                    try:
                        # Summarize result
                        if isinstance(result, list):
                            span.end(output={"count": len(result)})
                        elif isinstance(result, str):
                            span.end(output={"length": len(result)})
                        elif isinstance(result, dict):
                            span.end(output={"keys": list(result.keys())})
                        else:
                            span.end()
                    except Exception:
                        pass
                return result
            except Exception as e:
                if span:
                    try:
                        span.end(output={"error": str(e)}, level="ERROR")
                    except Exception:
                        pass
                raise

        return wrapper
    return decorator


@asynccontextmanager
async def trace_full_pipeline(
    user_id: int,
    question: str,
    metadata: Optional[dict] = None,
):
    """Context manager for tracing the full RAG pipeline.

    Usage:
        async with trace_full_pipeline(tg_id, question) as trace:
            # Pipeline steps...
            trace.span("rewrite").end(output=rewritten)
            trace.generation("generate", "gemini").end(output=answer)
    """
    trace = TraceContext(
        name="rag_pipeline",
        user_id=str(user_id),
        metadata={
            "question": question[:200],
            **(metadata or {}),
        },
    )

    async with trace:
        yield trace


def log_pipeline_metrics(
    user_id: int,
    question: str,
    timings: dict,
    docs_retrieved: int,
    docs_graded: int,
    cache_hit: bool,
    error: Optional[str] = None,
):
    """Log pipeline metrics to LangFuse.

    Called at the end of pipeline execution for analytics.
    """
    langfuse = _get_langfuse()
    if not langfuse:
        return

    try:
        langfuse.score(
            name="pipeline_latency",
            value=timings.get("total", 0),
            data_type="NUMERIC",
            comment=f"User {user_id}",
        )

        if cache_hit:
            langfuse.event(
                name="cache_hit",
                metadata={"user_id": user_id, "question": question[:100]},
            )

        if error:
            langfuse.event(
                name="pipeline_error",
                metadata={"user_id": user_id, "error": error},
                level="ERROR",
            )

        langfuse.flush()
    except Exception as e:
        logger.debug(f"Failed to log metrics: {e}")


def log_routing_decision(
    user_id: int,
    question: str,
    intent: str,
    suggested_tools: list,
    confidence: float,
    reason: str,
    latency_ms: float = 0,
):
    """Log intent routing decision to LangFuse.

    Tracks distribution of intents for analytics and pattern optimization.

    Args:
        user_id: Telegram user ID
        question: User's question
        intent: Classified intent (tool_only, rag_only, tool_rag, chitchat)
        suggested_tools: List of suggested tool names
        confidence: Classification confidence (0-1)
        reason: Human-readable reason for classification
        latency_ms: Time taken to classify (milliseconds)
    """
    langfuse = _get_langfuse()
    if not langfuse:
        # Fallback to standard logging for local debugging
        logger.info(
            f"[ROUTING] user={user_id} intent={intent} confidence={confidence:.2f} "
            f"tools={suggested_tools} latency={latency_ms:.1f}ms reason={reason}"
        )
        return

    try:
        # Log as event with full metadata
        langfuse.event(
            name="routing_decision",
            metadata={
                "user_id": user_id,
                "question": question[:200],
                "intent": intent,
                "suggested_tools": suggested_tools,
                "confidence": confidence,
                "reason": reason,
                "latency_ms": latency_ms,
            },
        )

        # Log confidence as score for analytics
        langfuse.score(
            name="routing_confidence",
            value=confidence,
            data_type="NUMERIC",
            comment=f"Intent: {intent}",
        )

        langfuse.flush()
    except Exception as e:
        logger.debug(f"Failed to log routing decision: {e}")


# ── In-memory metrics for local stats ──────────────────────────────────

_routing_stats = {
    "tool_only": 0,
    "rag_only": 0,
    "tool_rag": 0,
    "chitchat": 0,
    "total": 0,
}


def increment_routing_stat(intent: str):
    """Increment in-memory routing counter."""
    global _routing_stats
    _routing_stats["total"] += 1
    if intent in _routing_stats:
        _routing_stats[intent] += 1


def get_routing_stats() -> dict:
    """Get current routing statistics.

    Returns:
        dict with counts and percentages for each intent type
    """
    total = _routing_stats["total"]
    if total == 0:
        return {
            "total": 0,
            "distribution": {},
            "percentages": {},
        }

    distribution = {k: v for k, v in _routing_stats.items() if k != "total"}
    percentages = {k: round(v / total * 100, 1) for k, v in distribution.items()}

    return {
        "total": total,
        "distribution": distribution,
        "percentages": percentages,
    }


def reset_routing_stats():
    """Reset routing statistics (useful for testing)."""
    global _routing_stats
    _routing_stats = {
        "tool_only": 0,
        "rag_only": 0,
        "tool_rag": 0,
        "chitchat": 0,
        "total": 0,
    }


def shutdown():
    """Shutdown LangFuse client gracefully."""
    global _langfuse_client
    if _langfuse_client:
        try:
            _langfuse_client.flush()
            _langfuse_client.shutdown()
        except Exception:
            pass
        _langfuse_client = None
