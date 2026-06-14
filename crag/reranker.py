"""Optional, API-based cross-encoder reranker.

This module is a thin, *optional* layer that reorders retrieved documents by
relevance to a query using an external rerank API (Cohere, Jina, or NVIDIA).

Architectural notes
--------------------
- This project is API-only: no ``torch``, no ``sentence-transformers``, and no
  local cross-encoder models are run in-process. Reranking, if enabled, is
  always delegated to a hosted rerank endpoint over HTTP via ``aiohttp``.
- The reranker is OFF by default. If ``RERANKER_PROVIDER`` is unset (or set
  to ``"none"``) or no API key is configured, :func:`reranker_enabled`
  returns ``False`` and :func:`rerank_documents` is a no-op that returns the
  first ``top_k`` documents unchanged, in their original order. This means a
  deployment without rerank configuration behaves exactly as it did before
  this module existed.
- Importing this module never performs any network I/O and never raises,
  even if ``aiohttp`` is missing or environment variables are unset.
- Every code path in :func:`rerank_documents` is wrapped in defensive
  error handling: any failure (missing dependency, network error, bad
  response shape, timeout, etc.) is logged with ``logger.warning`` and the
  function falls back to ``docs[:top_k]``. It never raises.

Configuration (environment variables)
--------------------------------------
- ``RERANKER_PROVIDER``: one of ``none`` (default), ``cohere``, ``jina``,
  ``nvidia``. Any other/unknown value is treated as disabled.
- ``RERANKER_API_KEY``: API key for the chosen provider. Required for the
  reranker to be considered enabled.
- ``RERANKER_MODEL``: optional model name override. If unset, each provider
  uses a sensible multilingual default (the bot fields questions in
  Russian, among other languages):
    - cohere:  ``rerank-v3.5``
    - jina:    ``jina-reranker-v2-base-multilingual``
    - nvidia:  ``nvidia/llama-3.2-nv-rerankqa-1b-v2``

Usage
-----
    from crag.reranker import reranker_enabled, rerank_documents

    if reranker_enabled():
        docs = await rerank_documents(query, candidate_docs, top_k=6)
    else:
        docs = candidate_docs[:6]

(In practice it is safe to *always* call :func:`rerank_documents` — it
already checks :func:`reranker_enabled` internally and degrades to the
identity slice when disabled.)
"""

from __future__ import annotations

import logging
import os
from typing import Any, List

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

# Supported provider identifiers.
_SUPPORTED_PROVIDERS = {"cohere", "jina", "nvidia"}

# Per-provider sensible (multilingual) default models.
_DEFAULT_MODELS = {
    "cohere": "rerank-v3.5",
    "jina": "jina-reranker-v2-base-multilingual",
    "nvidia": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
}

# Per-provider rerank endpoints.
_ENDPOINTS = {
    "cohere": "https://api.cohere.com/v2/rerank",
    "jina": "https://api.jina.ai/v1/rerank",
    "nvidia": "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
}

# Maximum number of characters of a document's text sent to the rerank API.
_MAX_DOC_CHARS = 4000

# HTTP timeout (seconds) for the rerank request.
_HTTP_TIMEOUT_SECONDS = 10


def _get_provider() -> str:
    """Return the configured provider name, lowercased and stripped."""
    return os.environ.get("RERANKER_PROVIDER", "none").strip().lower()


def _get_api_key() -> str:
    """Return the configured API key (empty string if unset)."""
    return os.environ.get("RERANKER_API_KEY", "").strip()


def _get_model(provider: str) -> str:
    """Return the configured model override or the provider's default."""
    override = os.environ.get("RERANKER_MODEL", "").strip()
    if override:
        return override
    return _DEFAULT_MODELS.get(provider, "")


def reranker_enabled() -> bool:
    """Return True only if a supported provider AND an API key are configured.

    This is a pure, side-effect-free check of environment variables. It never
    raises and never performs I/O.
    """
    provider = _get_provider()
    if provider not in _SUPPORTED_PROVIDERS:
        return False
    if not _get_api_key():
        return False
    return True


def _doc_text(doc: Any) -> str:
    """Extract and truncate the text of a ``crag.simple_rag.Document``.

    Falls back gracefully if the object doesn't have ``page_content``.
    """
    text = getattr(doc, "page_content", None)
    if text is None:
        text = str(doc)
    text = str(text)
    if len(text) > _MAX_DOC_CHARS:
        text = text[:_MAX_DOC_CHARS]
    return text


async def _call_cohere(
    session: Any, query: str, texts: List[str], model: str, api_key: str, top_n: int
) -> List[dict]:
    """Call Cohere's rerank API and return its ``results`` list.

    Each result item is expected to look like::

        {"index": <int>, "relevance_score": <float>}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "query": query,
        "documents": texts,
        "top_n": top_n,
    }
    async with session.post(_ENDPOINTS["cohere"], json=payload, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data.get("results", [])


async def _call_jina(
    session: Any, query: str, texts: List[str], model: str, api_key: str, top_n: int
) -> List[dict]:
    """Call Jina AI's rerank API and return its ``results`` list.

    Each result item is expected to look like::

        {"index": <int>, "relevance_score": <float>}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "query": query,
        "documents": texts,
        "top_n": top_n,
    }
    async with session.post(_ENDPOINTS["jina"], json=payload, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data.get("results", [])


async def _call_nvidia(
    session: Any, query: str, texts: List[str], model: str, api_key: str, top_n: int
) -> List[dict]:
    """Call an NVIDIA NIM reranking endpoint and return a results list.

    NVIDIA's NIM "ranking" endpoints generally accept a payload shaped like::

        {
            "model": "...",
            "query": {"text": "..."},
            "passages": [{"text": "..."}, ...],
        }

    and return::

        {"rankings": [{"index": <int>, "logit": <float>}, ...]}

    This is implemented best-effort: it is normalized to the same
    ``[{"index": int, "relevance_score": float}, ...]`` shape used by the
    other providers. If NVIDIA changes/varies this schema, any parsing error
    is caught by the caller and results in a graceful fallback.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "query": {"text": query},
        "passages": [{"text": t} for t in texts],
    }
    async with session.post(_ENDPOINTS["nvidia"], json=payload, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()

    raw_results = data.get("rankings", data.get("results", []))
    normalized: List[dict] = []
    for item in raw_results:
        if "relevance_score" in item:
            score = item["relevance_score"]
        else:
            score = item.get("logit", 0.0)
        normalized.append({"index": item["index"], "relevance_score": score})

    # Sort by score descending and trim to top_n, mirroring Cohere/Jina which
    # already return results pre-sorted and limited to top_n.
    normalized.sort(key=lambda r: r.get("relevance_score", 0.0), reverse=True)
    return normalized[:top_n]


_PROVIDER_CALLERS = {
    "cohere": _call_cohere,
    "jina": _call_jina,
    "nvidia": _call_nvidia,
}


async def rerank_documents(query: str, docs: list, top_k: int = 6) -> list:
    """Rerank ``docs`` by relevance to ``query`` via an external rerank API.

    Parameters
    ----------
    query:
        The user's query text.
    docs:
        A list of ``crag.simple_rag.Document`` instances (or any object with
        a ``page_content`` attribute).
    top_k:
        Maximum number of documents to return.

    Returns
    -------
    list
        The top ``top_k`` documents, reordered by descending relevance
        according to the configured rerank API.

    This function NEVER raises. If the reranker is disabled (see
    :func:`reranker_enabled`), if ``docs`` is empty, or if anything goes
    wrong while calling the rerank API (missing ``aiohttp``, network error,
    timeout, unexpected response shape, etc.), it logs a warning and falls
    back to ``docs[:top_k]`` in the original order.
    """
    if not docs:
        return docs[:top_k]

    if not reranker_enabled():
        return docs[:top_k]

    try:
        import aiohttp  # Lazy import: keep this module importable even
        # in environments where aiohttp is not installed, and avoid any
        # import-time dependency requirements.
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Reranker disabled: aiohttp unavailable (%s)", exc)
        return docs[:top_k]

    provider = _get_provider()
    api_key = _get_api_key()
    model = _get_model(provider)
    caller = _PROVIDER_CALLERS.get(provider)

    if caller is None:  # pragma: no cover - reranker_enabled() already guards this
        logger.warning("Reranker disabled: unsupported provider %r", provider)
        return docs[:top_k]

    texts = [_doc_text(doc) for doc in docs]
    top_n = min(top_k, len(docs))

    try:
        timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            results = await caller(session, query, texts, model, api_key, top_n)

        reranked: List[Any] = []
        seen_indices = set()
        for item in results:
            idx = item.get("index")
            if idx is None:
                continue
            idx = int(idx)
            if idx < 0 or idx >= len(docs) or idx in seen_indices:
                continue
            seen_indices.add(idx)
            reranked.append(docs[idx])
            if len(reranked) >= top_k:
                break

        if not reranked:
            logger.warning(
                "Reranker (%s) returned no usable results; falling back to original order",
                provider,
            )
            return docs[:top_k]

        return reranked
    except Exception as exc:
        logger.warning("Reranker (%s) request failed: %s", provider, exc)
        return docs[:top_k]
