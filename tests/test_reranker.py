"""
Unit tests for crag.reranker (optional, API-based cross-encoder reranker).

These tests use NO real network calls. The aiohttp HTTP layer is mocked out
by injecting a fake ``aiohttp`` module into ``sys.modules`` (or by
monkeypatching the provider-specific ``_call_*`` helpers directly), so the
tests can run with just pytest + stdlib even if the real ``aiohttp`` package
is not installed in the test environment.

Run with:  python -m pytest tests/test_reranker.py -v
"""

import asyncio
import sys
import types

import pytest

from crag import reranker
from crag.simple_rag import Document


# ── Helpers ──────────────────────────────────────────────────────────────


def make_docs():
    return [
        Document(page_content="Document about admission deadlines.", metadata={"id": "a"}),
        Document(page_content="Document about dormitory and housing.", metadata={"id": "b"}),
        Document(page_content="Document about scholarships and tuition.", metadata={"id": "c"}),
        Document(page_content="Document about visa requirements.", metadata={"id": "d"}),
    ]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure reranker-related env vars start unset for every test."""
    for var in ("RERANKER_PROVIDER", "RERANKER_API_KEY", "RERANKER_MODEL"):
        monkeypatch.delenv(var, raising=False)
    yield


# ── (a) reranker_enabled() is False when env unset ─────────────────────────


def test_reranker_enabled_false_by_default():
    assert reranker.reranker_enabled() is False


def test_reranker_enabled_false_with_provider_none():
    import os

    os.environ["RERANKER_PROVIDER"] = "none"
    os.environ["RERANKER_API_KEY"] = "some-key"
    try:
        assert reranker.reranker_enabled() is False
    finally:
        del os.environ["RERANKER_PROVIDER"]
        del os.environ["RERANKER_API_KEY"]


def test_reranker_enabled_false_without_api_key(monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    # No RERANKER_API_KEY set
    assert reranker.reranker_enabled() is False


def test_reranker_enabled_true_with_provider_and_key(monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")
    assert reranker.reranker_enabled() is True


def test_reranker_enabled_false_for_unsupported_provider(monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "some-unsupported-provider")
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")
    assert reranker.reranker_enabled() is False


# ── (b) rerank_documents returns docs[:top_k] unchanged when disabled ─────


def test_rerank_documents_disabled_returns_prefix_unchanged():
    docs = make_docs()
    result = run(reranker.rerank_documents("some query", docs, top_k=2))
    assert result == docs[:2]


def test_rerank_documents_disabled_with_empty_docs():
    result = run(reranker.rerank_documents("some query", [], top_k=6))
    assert result == []


# ── (c) reorders correctly and respects top_k when HTTP call is mocked ────


def test_rerank_documents_reorders_with_mocked_call(monkeypatch):
    docs = make_docs()

    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")

    # Pretend the rerank API ranked doc index 2 highest, then 0, then 3,
    # then 1 (descending relevance). Results are already capped to top_n by
    # the (fake) API.
    fake_results = [
        {"index": 2, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.80},
        {"index": 3, "relevance_score": 0.10},
    ]

    async def fake_caller(session, query, texts, model, api_key, top_n):
        assert query == "some query"
        assert len(texts) == len(docs)
        assert top_n == 3
        return fake_results[:top_n]

    monkeypatch.setitem(reranker._PROVIDER_CALLERS, "cohere", fake_caller)

    # Provide a minimal fake aiohttp module so the lazy `import aiohttp`
    # inside rerank_documents succeeds without the real dependency.
    fake_aiohttp = _make_fake_aiohttp()
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    result = run(reranker.rerank_documents("some query", docs, top_k=3))

    assert result == [docs[2], docs[0], docs[3]]
    assert len(result) == 3


def test_rerank_documents_respects_top_k_smaller_than_results(monkeypatch):
    docs = make_docs()

    monkeypatch.setenv("RERANKER_PROVIDER", "jina")
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")

    fake_results = [
        {"index": 1, "relevance_score": 0.9},
        {"index": 3, "relevance_score": 0.7},
        {"index": 0, "relevance_score": 0.5},
        {"index": 2, "relevance_score": 0.1},
    ]

    async def fake_caller(session, query, texts, model, api_key, top_n):
        return fake_results

    monkeypatch.setitem(reranker._PROVIDER_CALLERS, "jina", fake_caller)

    fake_aiohttp = _make_fake_aiohttp()
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    result = run(reranker.rerank_documents("query", docs, top_k=2))

    assert result == [docs[1], docs[3]]
    assert len(result) == 2


# ── (d) on mocked HTTP error, falls back to docs[:top_k] without raising ──


def test_rerank_documents_falls_back_on_caller_exception(monkeypatch):
    docs = make_docs()

    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")

    async def failing_caller(session, query, texts, model, api_key, top_n):
        raise RuntimeError("simulated network failure")

    monkeypatch.setitem(reranker._PROVIDER_CALLERS, "cohere", failing_caller)

    fake_aiohttp = _make_fake_aiohttp()
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    result = run(reranker.rerank_documents("query", docs, top_k=2))

    assert result == docs[:2]


def test_rerank_documents_falls_back_when_aiohttp_missing(monkeypatch):
    docs = make_docs()

    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")

    # Simulate aiohttp not being installed at all.
    monkeypatch.setitem(sys.modules, "aiohttp", None)

    result = run(reranker.rerank_documents("query", docs, top_k=2))

    assert result == docs[:2]


def test_rerank_documents_falls_back_on_empty_results(monkeypatch):
    docs = make_docs()

    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")

    async def empty_caller(session, query, texts, model, api_key, top_n):
        return []

    monkeypatch.setitem(reranker._PROVIDER_CALLERS, "cohere", empty_caller)

    fake_aiohttp = _make_fake_aiohttp()
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    result = run(reranker.rerank_documents("query", docs, top_k=2))

    assert result == docs[:2]


# ── Fake aiohttp module for tests that exercise the ClientSession path ────


def _make_fake_aiohttp():
    """Build a minimal fake ``aiohttp`` module sufficient for our code path.

    The real rerank_documents() only uses ``aiohttp.ClientTimeout`` and
    ``aiohttp.ClientSession`` as an async context manager. The actual HTTP
    POST is performed inside the (mocked) provider ``_call_*`` callers, so
    the fake session never needs to issue real requests.
    """

    class _FakeClientTimeout:
        def __init__(self, total=None):
            self.total = total

    class _FakeClientSession:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    fake_module = types.ModuleType("aiohttp")
    fake_module.ClientTimeout = _FakeClientTimeout
    fake_module.ClientSession = _FakeClientSession
    return fake_module


# ── Model / config defaults ────────────────────────────────────────────────


def test_default_models_are_multilingual(monkeypatch):
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    assert reranker._get_model("cohere") == "rerank-v3.5"
    assert "multilingual" in reranker._get_model("jina")


def test_model_override_env_var(monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", "custom-model")
    assert reranker._get_model("cohere") == "custom-model"
    assert reranker._get_model("jina") == "custom-model"


def test_doc_text_truncates_long_text():
    long_text = "x" * 5000
    doc = Document(page_content=long_text, metadata={})
    text = reranker._doc_text(doc)
    assert len(text) == reranker._MAX_DOC_CHARS
