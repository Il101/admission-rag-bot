import os
import json
import logging
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, AsyncGenerator

from jinja2 import BaseLoader, Environment, Undefined
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker as sa_sessionmaker
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ── Embedding cache (LRU, thread-safe via asyncio single-thread) ────────

_EMBEDDING_CACHE_MAX = 256


class _LRUCache(OrderedDict):
    """Simple LRU cache backed by OrderedDict."""

    def __init__(self, maxsize: int):
        super().__init__()
        self._maxsize = maxsize

    def get_or_none(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self._maxsize:
            self.popitem(last=False)


_embedding_cache = _LRUCache(_EMBEDDING_CACHE_MAX)


# ── Semantic answer cache ────────────────────────────────────────────────


class _SemanticCache:
    """Cache full pipeline answers keyed by question embedding similarity.

    If a new question's embedding cosine similarity to a cached entry
    exceeds *threshold*, the cached answer JSON is returned — skipping
    retrieval, grading, and generation entirely.
    """

    def __init__(self, max_size: int = 128, threshold: float = 0.97):
        self._entries: list[tuple[list[float], str]] = []
        self._max_size = max_size
        self._threshold = threshold

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def get(self, embedding: list[float]) -> str | None:
        best_sim, best_answer = 0.0, None
        for cached_emb, cached_answer in self._entries:
            sim = self._cosine_sim(embedding, cached_emb)
            if sim > best_sim:
                best_sim, best_answer = sim, cached_answer
        if best_sim >= self._threshold:
            logger.info("Semantic cache HIT (sim=%.4f)", best_sim)
            return best_answer
        return None

    def put(self, embedding: list[float], answer: str):
        if len(self._entries) >= self._max_size:
            self._entries.pop(0)
        self._entries.append((embedding, answer))


_answer_cache = _SemanticCache()


# ── Jinja2 safe template rendering ──────────────────────────────────────


class _SilentUndefined(Undefined):
    """Jinja2 Undefined that renders as empty string instead of raising."""

    def __str__(self):
        return ""

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False


_jinja_env = Environment(loader=BaseLoader(), undefined=_SilentUndefined)


def _render_template(template_str: str, **kwargs) -> str:
    """Render a prompt template with Jinja2.

    Handles both Jinja2 ``{{ var }}`` and legacy ``{var}`` syntax.
    Unknown variables are rendered as empty strings instead of raising.
    """
    # Convert legacy {var} placeholders to Jinja2 {{ var }} syntax
    # but skip already-doubled {{ and JSON braces like {"key": ...}
    import re

    def _convert_legacy(tpl: str) -> str:
        # Replace {word} but NOT {{word}} and NOT {"...
        return re.sub(
            r'(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})',
            r'{{ \1 }}',
            tpl,
        )

    try:
        converted = _convert_legacy(template_str)
        tpl = _jinja_env.from_string(converted)
        return tpl.render(**kwargs)
    except Exception:
        # Fallback to naive replace for backward compat
        result = template_str
        for k, v in kwargs.items():
            result = result.replace("{" + k + "}", str(v))
        return result


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class Document:
    page_content: str
    metadata: dict


def documents_to_context_str(docs: List[Document]) -> str:
    """Convert documents to context string with source attribution."""
    parts = []
    for doc in docs:
        source_url = doc.metadata.get("source_url", "")
        section = doc.metadata.get("section_path", "")
        title = doc.metadata.get("title", doc.metadata.get("source", ""))

        # Build attribution header
        header_parts = []
        if title:
            header_parts.append(title)
        if section:
            header_parts.append(section)
        if source_url:
            header_parts.append(source_url)

        header = " — ".join(header_parts)
        if header:
            parts.append(f"[Источник: {header}]\n{doc.page_content}")
        else:
            parts.append(doc.page_content)

    return "\n\n".join(parts)


# ── Retry helper ─────────────────────────────────────────────────────────


async def retry_on_503(coro_func, *args, max_retries=3, initial_delay=1, **kwargs):
    """Simple exponential backoff retry for 503/429 errors."""
    for attempt in range(max_retries):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            is_503 = "503" in err_msg or "unavailable" in err_msg
            is_429 = "429" in err_msg or "exhausted" in err_msg

            if is_503 or is_429:
                if attempt == max_retries - 1:
                    raise
                delay = initial_delay * (2 ** attempt)
                logger.warning(
                    f"Gemini API error (attempt {attempt+1}/{max_retries}): {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                raise


# ── Main RAG class ───────────────────────────────────────────────────────


class SimpleRAG:
    """A direct implementation of RAG using google-genai async client
    and async SQLAlchemy for non-blocking DB operations.

    Features:
    - Async DB via SQLAlchemy AsyncEngine (no thread-pool blocking)
    - Hybrid search: vector similarity + PostgreSQL full-text search (BM25-like)
    - Batch document grading in a single LLM call
    - LRU cache for embeddings
    - Jinja2 prompt templating (safe against injection)
    """

    # ── Hybrid search weight (0.0 = only vector, 1.0 = only FTS) ────────
    FTS_WEIGHT = 0.3
    VECTOR_WEIGHT = 0.7

    def __init__(self, db_engine, prompts_config: dict = None):
        self.api_key = os.environ.get(
            "GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY")
        )
        self.client = genai.Client(api_key=self.api_key)

        # Keep sync engine for backward-compat (used by indexing script)
        self.db_engine = db_engine

        # Build async engine from the sync engine's URL
        sync_url = db_engine.url.render_as_string(hide_password=False)
        async_url = self._to_async_url(sync_url)
        self._async_engine = create_async_engine(
            async_url, pool_size=5, max_overflow=10
        )
        self._async_session_factory = sa_sessionmaker(
            self._async_engine, class_=AsyncSession, expire_on_commit=False
        )

        self.rag_prompt_messages = []
        self.system_prompt = ""
        self.human_prompt_template = (
            "{question}\nКонтекст:\n{context}"
        )

        if prompts_config:
            self.rag_prompt_messages = (
                prompts_config.get("rag_prompt", {}).get("messages", [])
            )
            self.system_prompt = self._extract_system_prompt(
                self.rag_prompt_messages
            )
            self.human_prompt_template = self._extract_human_prompt(
                self.rag_prompt_messages
            )

            # Load rewriting prompt template
            rewriting_messages = (
                prompts_config.get("rewriting_prompt", {}).get("messages", [])
            )
            self.rewrite_prompt_template = (
                self._extract_human_prompt_from_messages(rewriting_messages)
            )
            self.rewrite_system_prompt = self._extract_system_prompt(
                rewriting_messages
            )

            # Load grading prompt template
            grading_messages = (
                prompts_config.get("grading_prompt", {}).get("messages", [])
            )
            self.grading_prompt_template = (
                self._extract_human_prompt_from_messages(grading_messages)
            )
            self.grading_system_prompt = self._extract_system_prompt(
                grading_messages
            )

    # ── URL conversion ───────────────────────────────────────────────────

    @staticmethod
    def _to_async_url(sync_url: str) -> str:
        """Convert a sync SQLAlchemy URL to an async one (psycopg → asyncpg)."""
        url = sync_url
        if "+psycopg" in url:
            url = url.replace("+psycopg", "+asyncpg")
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    # ── Prompt extraction ────────────────────────────────────────────────

    def _extract_system_prompt(self, messages: list) -> str:
        for m in messages:
            if m.get("_target_") == "langchain_core.messages.SystemMessage":
                return m.get("content", "")
        return "Ты — полезный ассистент."

    def _extract_human_prompt(self, messages: list) -> str:
        for m in messages:
            if m.get("_target_") == (
                "langchain_core.prompts."
                "HumanMessagePromptTemplate.from_template"
            ):
                return m.get("template", "")
        return (
            "Вопрос: {question}\nКонтекст: {context}\n"
            "История: {chat_history}\nПамять: {memory_context}"
        )

    def _extract_human_prompt_from_messages(self, messages: list) -> str:
        return self._extract_human_prompt(messages)

    # ── Embedding with cache ─────────────────────────────────────────────

    async def get_embedding(self, text_input: str) -> List[float]:
        if not text_input.strip():
            return []

        # Check cache first
        cache_key = hash(text_input[:512])
        cached = _embedding_cache.get_or_none(cache_key)
        if cached is not None:
            return cached

        try:
            response = await retry_on_503(
                self.client.aio.models.embed_content,
                model="models/gemini-embedding-001",
                contents=text_input,
            )
            embedding = response.embeddings[0].values
            _embedding_cache.put(cache_key, embedding)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []

    # ── Filter SQL builders ──────────────────────────────────────────────

    @staticmethod
    def _build_filter_sql(user_filters: dict) -> tuple[str, dict]:
        conditions = []
        params = {}

        country = user_filters.get("country_scope")
        if country and country != "ALL":
            conditions.append(
                "(metadata->'country_scope' @> '\"ALL\"'::jsonb "
                "OR metadata->'country_scope' @> (:country_filter)::jsonb)"
            )
            params["country_filter"] = json.dumps(country)

        where_clause = (
            ("WHERE " + " AND ".join(conditions)) if conditions else ""
        )
        return where_clause, params

    def _build_level_boost_expr(
        self, user_filters: dict
    ) -> tuple[str, dict]:
        level = user_filters.get("level")
        if not level:
            return "", {}
        return (
            "CASE WHEN metadata->'level' @> (:user_level)::jsonb "
            "THEN 0 ELSE 1 END",
            {"user_level": json.dumps([level])},
        )

    # ── Async DB retrieval (hybrid: vector + FTS) ────────────────────────

    async def aretrieve(
        self, query: str, top_k: int = 6, user_filters: dict = None
    ) -> List[Document]:
        embedding = await self.get_embedding(query)
        if not embedding:
            return []

        where_clause = ""
        filter_params = {}
        level_boost_expr = ""
        level_params = {}

        if user_filters:
            where_clause, filter_params = self._build_filter_sql(user_filters)
            level_boost_expr, level_params = self._build_level_boost_expr(
                user_filters
            )

        # Hybrid scoring: combine vector cosine distance and FTS rank
        # Lower score = better match
        vector_score = "embedding <=> CAST(:embedding AS vector)"
        fts_score = (
            "COALESCE(1.0 - ts_rank_cd("
            "to_tsvector('simple', content) || to_tsvector('russian', content), "
            "plainto_tsquery('simple', :query_text) || plainto_tsquery('russian', :query_text)"
            "), 1.0)"
        )
        hybrid_score = (
            f"({self.VECTOR_WEIGHT} * {vector_score} + "
            f"{self.FTS_WEIGHT} * {fts_score})"
        )

        if level_boost_expr:
            order_by = f"ORDER BY {level_boost_expr} ASC, {hybrid_score}"
        else:
            order_by = f"ORDER BY {hybrid_score}"

        sql_str = f"""
            SELECT content, metadata
            FROM simple_documents
            {where_clause}
            {order_by}
            LIMIT :top_k
        """

        all_params = {
            "embedding": str(embedding),
            "query_text": query,
            "top_k": top_k,
            **filter_params,
            **level_params,
        }

        # Use async engine — no thread-pool blocking
        async with self._async_session_factory() as session:
            result = await session.execute(text(sql_str), all_params)
            docs = [
                Document(page_content=row[0], metadata=row[1])
                for row in result
            ]
        return docs

    # ── Query rewriting ──────────────────────────────────────────────────

    async def arewrite_query(
        self, question: str, chat_history: str, memory_context: str
    ) -> str:
        if not getattr(self, "rewrite_prompt_template", None):
            return question

        user_prompt = _render_template(
            self.rewrite_prompt_template,
            question=question,
            chat_history=chat_history,
            memory_context=memory_context,
        )

        try:
            response = await retry_on_503(
                self.client.aio.models.generate_content,
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.rewrite_system_prompt,
                    temperature=0.0,
                ),
            )
            rewritten = (response.text or "").strip()
            if rewritten:
                logger.info(
                    f"Query rewrite: '{question[:60]}' → '{rewritten[:60]}'"
                )
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")
        return question

    # ── Batch document grading (single LLM call instead of N) ────────────

    async def agrade_documents(
        self, question: str, docs: List[Document]
    ) -> List[Document]:
        if not getattr(self, "grading_prompt_template", None) or not docs:
            return docs

        # Build a single batch prompt listing all documents
        docs_listing = ""
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:600].replace("\n", " ")
            docs_listing += f"[DOC_{i}] {snippet}\n\n"

        batch_prompt = (
            f"ВОПРОС: {question}\n\n"
            f"Ниже приведены {len(docs)} фактов/документов. "
            "Для КАЖДОГО укажи, релевантен ли он вопросу.\n\n"
            f"{docs_listing}"
            "Ответь строго в формате JSON: "
            '{"scores": [1, 0, 1, ...]} '
            "где 1 = релевантен, 0 = нерелевантен. "
            f"Массив должен содержать ровно {len(docs)} элементов."
        )

        try:
            response = await retry_on_503(
                self.client.aio.models.generate_content,
                model="gemini-2.5-flash",
                contents=batch_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.grading_system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            raw = (response.text or "{}").strip()
            data = json.loads(raw)
            scores = data.get("scores", [1] * len(docs))

            # Validate length
            if len(scores) != len(docs):
                logger.warning(
                    f"Grading returned {len(scores)} scores for "
                    f"{len(docs)} docs, keeping all."
                )
                return docs

            relevant = [
                doc for doc, score in zip(docs, scores) if int(score) == 1
            ]

            if relevant:
                logger.info(
                    f"Grading: {len(docs)} → {len(relevant)} relevant docs"
                )
                return relevant
            else:
                logger.info(
                    f"Grading filtered all docs, falling back to full set "
                    f"({len(docs)})"
                )
                return docs

        except Exception as e:
            logger.warning(f"Batch grading failed, keeping all docs: {e}")
            return docs

    # ── Streaming answer ─────────────────────────────────────────────────

    async def astream_answer(
        self,
        question: str,
        context: str,
        chat_history: str,
        memory_context: str,
        current_date: str,
    ) -> AsyncGenerator[str, None]:
        sys_prompt = _render_template(
            self.system_prompt, current_date=current_date
        )
        user_prompt = _render_template(
            self.human_prompt_template,
            question=question,
            memory_context=memory_context,
            chat_history=chat_history,
            context=context,
        )

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "answer": {
                    "type": "STRING",
                    "description": (
                        "The main text of the response to the user question."
                    ),
                },
                "suggested_questions": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": (
                        "3-4 suggested follow-up questions relevant "
                        "to the current state."
                    ),
                },
            },
            "required": ["answer", "suggested_questions"],
        }

        try:
            stream = await retry_on_503(
                self.client.aio.models.generate_content_stream,
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )

            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Error streaming from Gemini: {e}")
            yield json.dumps(
                {
                    "answer": (
                        "Извините, произошла ошибка при генерации ответа. "
                        "Пожалуйста, попробуйте еще раз через минуту."
                    ),
                    "suggested_questions": [],
                }
            )

    # ── Semantic answer cache ────────────────────────────────────────────

    async def get_cached_answer(self, question: str) -> str | None:
        """Check if a semantically similar question was already answered."""
        emb = await self.get_embedding(question)
        if not emb:
            return None
        return _answer_cache.get(emb)

    async def cache_answer(self, question: str, answer_json: str):
        """Store a question → answer pair in the semantic cache."""
        emb = await self.get_embedding(question)
        if emb:
            _answer_cache.put(emb, answer_json)

    # ── LLM-based summary compression ───────────────────────────────────

    async def acompress_summary(self, summary: str) -> str:
        """Use LLM to compress a conversation summary, keeping key facts."""
        prompt = (
            "Сожми следующее резюме разговора с ботом о поступлении в Австрию "
            "в 5-8 ключевых фактов о пользователе. "
            "Сохрани: конкретные вузы, города, уровни языка, сроки, документы, "
            "принятые решения. Убери: формулировки вопросов и общие фразы.\n"
            "Формат: простой текст, каждый факт с новой строки.\n\n"
            f"{summary}"
        )
        try:
            response = await retry_on_503(
                self.client.aio.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            compressed = (response.text or "").strip()
            if compressed and len(compressed) < len(summary):
                logger.info(
                    "Summary compressed: %d → %d chars", len(summary), len(compressed)
                )
                return compressed
        except Exception as e:
            logger.warning("Summary compression failed: %s", e)
        return summary

    # ── Document management ──────────────────────────────────────────────

    async def aadd_documents(self, docs: List[Document]) -> List[int]:
        ids = []
        for doc in docs:
            embedding = await self.get_embedding(doc.page_content)
            if not embedding:
                continue
            async with self._async_session_factory() as session:
                async with session.begin():
                    res = await session.execute(
                        text(
                            "INSERT INTO simple_documents "
                            "(content, metadata, embedding) "
                            "VALUES (:c, :m, CAST(:e AS vector)) "
                            "RETURNING id"
                        ),
                        {
                            "c": doc.page_content,
                            "m": json.dumps(doc.metadata),
                            "e": str(embedding),
                        },
                    )
                    ids.append(res.scalar())
        return ids

    async def adelete_documents(self, ids: List[str]) -> bool:
        try:
            async with self._async_session_factory() as session:
                async with session.begin():
                    for doc_id in ids:
                        await session.execute(
                            text(
                                "DELETE FROM simple_documents WHERE id = :id"
                            ),
                            {"id": int(doc_id)},
                        )
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    # ── Generic JSON generation ──────────────────────────────────────────

    async def agenerate_json(self, prompt: str) -> str:
        try:
            response = await retry_on_503(
                self.client.aio.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating JSON: {e}")
            return "{}"
