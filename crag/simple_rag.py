import os
import json
import logging
import asyncio
import hashlib
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
    """Simple exponential backoff retry for transient Gemini API errors.

    Retries on 503/429 (server overload) and connection-level failures
    (DNS resolution, TCP timeouts) that are common on Railway.
    """
    for attempt in range(max_retries):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            is_503 = "503" in err_msg or "unavailable" in err_msg
            is_429 = "429" in err_msg or "exhausted" in err_msg
            is_conn = (
                "clientconnector" in err_msg
                or "dns" in err_msg
                or "timed out" in err_msg
                or "connectionerror" in err_msg
            )

            if is_503 or is_429 or is_conn:
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

    # ── Embedding dimension settings ──────────────────────────────────────
    # Gemini embeddings are 3072 dims, but pgvector HNSW only supports up to 2000
    # Truncating to 1536 enables HNSW indexing while preserving most semantic info
    EMBEDDING_FULL_DIMS = 3072
    EMBEDDING_TRUNCATE_DIMS = 1536  # Set to None to disable truncation

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
        base_prompt = "Ты — полезный ассистент."
        for m in messages:
            if m.get("_target_") == "langchain_core.messages.SystemMessage":
                base_prompt = m.get("content", "")
                break
        
        # Add a strict anti-hallucination instruction for markdown tables
        anti_hallucination = (
            "\n\nКРИТИЧЕСКИ ВАЖНО: При чтении таблиц или списков в формате Markdown "
            "строго следи за тем, к какой сущности (городу, вузу) относится цена или дата. "
            "Никогда не смешивай данные из соседних строк таблицы! Например, если спрашивают про Инсбрук, "
            "не бери цену из строки для Линца."
        )
        return base_prompt + anti_hallucination

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

    async def get_embedding(
        self, text_input: str, truncate: bool = True
    ) -> List[float]:
        """Get embedding for text, optionally truncated for HNSW support.

        Args:
            text_input: Text to embed
            truncate: If True and EMBEDDING_TRUNCATE_DIMS is set, truncate embedding

        Returns:
            List of floats (embedding vector)
        """
        if not text_input.strip():
            return []

        # Check cache first (use full text SHA256 to avoid collisions)
        cache_key = hashlib.sha256(text_input.encode('utf-8')).hexdigest()
        if truncate and self.EMBEDDING_TRUNCATE_DIMS:
            cache_key = f"{cache_key}_t{self.EMBEDDING_TRUNCATE_DIMS}"

        cached = _embedding_cache.get_or_none(cache_key)
        if cached is not None:
            return cached

        try:
            response = await retry_on_503(
                self.client.aio.models.embed_content,
                model="models/gemini-embedding-001",
                contents=text_input,
            )
            embedding = list(response.embeddings[0].values)

            # Truncate for HNSW index support if enabled
            if truncate and self.EMBEDDING_TRUNCATE_DIMS:
                embedding = embedding[:self.EMBEDDING_TRUNCATE_DIMS]

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

    # ── HyDE (Hypothetical Document Embeddings) ────────────────────────

    async def agenerate_hypothetical_doc(
        self, question: str, memory_context: str = ""
    ) -> str:
        """Generate a hypothetical document that would answer the question.

        This improves retrieval by creating an embedding that's closer to
        the actual documents in the vector space.
        """
        context_hint = ""
        if memory_context:
            context_hint = f"\nКонтекст пользователя: {memory_context[:300]}"

        prompt = f"""Представь, что ты пишешь идеальный фрагмент из базы знаний
о поступлении в Австрию, который полностью отвечает на вопрос.

Вопрос: {question}{context_hint}

Напиши 2-3 информативных предложения как будто это выдержка из официального документа.
НЕ отвечай на вопрос напрямую — просто напиши, как выглядел бы идеальный документ с ответом.
Используй конкретные термины, даты, названия."""

        try:
            response = await retry_on_503(
                self.client.aio.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            hypothetical = (response.text or "").strip()
            if hypothetical:
                logger.info(
                    f"HyDE: '{question[:40]}...' → '{hypothetical[:60]}...'"
                )
                return hypothetical
        except Exception as e:
            logger.warning(f"HyDE generation failed, using original query: {e}")
        return question

    # ── Async DB retrieval (hybrid: vector + FTS) ────────────────────────

    async def aretrieve(
        self, query: str, top_k: int = 6, user_filters: dict = None,
        use_hyde: bool = False, memory_context: str = ""
    ) -> List[Document]:
        """Retrieve documents using hybrid search (vector + FTS).

        Args:
            query: The search query
            top_k: Number of documents to retrieve
            user_filters: Metadata filters (country_scope, level)
            use_hyde: If True, generate hypothetical document for better embedding
            memory_context: User context for HyDE generation
        """
        search_query = query
        if use_hyde:
            search_query = await self.agenerate_hypothetical_doc(query, memory_context)

        embedding = await self.get_embedding(search_query)
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
        # NOTE: parentheses around <=> are critical — without them
        # PostgreSQL parses "0.7 * embedding <=> ..." as "(0.7 * embedding) <=> ..."
        vector_score = "(embedding <=> CAST(:embedding AS vector))"
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

    # ── Query Decomposition ───────────────────────────────────────────────

    async def adecompose_query(self, question: str) -> list[str]:
        """Decompose a complex question into sub-questions.

        Complex questions with multiple aspects are split for parallel retrieval.
        Simple questions are returned as-is.

        Returns:
            List of sub-questions (1 element if question is simple)
        """
        prompt = f"""Проанализируй вопрос о поступлении в Австрию.
Если вопрос СЛОЖНЫЙ (содержит несколько разных аспектов), разбей его на 2-3 простых под-вопроса.
Если вопрос ПРОСТОЙ или касается одной темы, верни его без изменений.

ПРИМЕРЫ:
- "Какие документы нужны и какие дедлайны для TU Wien?" → ["Какие документы нужны для TU Wien?", "Какие дедлайны подачи в TU Wien?"]
- "Как подать документы в Uni Wien?" → ["Как подать документы в Uni Wien?"]
- "Сравни стоимость жизни в Вене и Граце, и какие там главные вузы?" → ["Стоимость жизни в Вене", "Стоимость жизни в Граце", "Главные вузы Вены и Граца"]

Вопрос: {question}

Ответь СТРОГО в JSON формате:
{{"sub_questions": ["вопрос1", "вопрос2", ...]}}

Если вопрос простой:
{{"sub_questions": ["{question}"]}}"""

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
            data = json.loads(response.text or "{}")
            sub_questions = data.get("sub_questions", [question])

            if len(sub_questions) > 1:
                logger.info(
                    f"Query decomposed: '{question[:50]}' → {len(sub_questions)} sub-questions"
                )
            return sub_questions if sub_questions else [question]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [question]

    async def aretrieve_decomposed(
        self,
        question: str,
        top_k: int = 6,
        user_filters: dict = None,
        use_hyde: bool = False,
        memory_context: str = "",
    ) -> List[Document]:
        """Retrieve with query decomposition for complex questions.

        Decomposes the question, retrieves for each sub-question in parallel,
        then merges and deduplicates results.
        """
        sub_questions = await self.adecompose_query(question)

        if len(sub_questions) <= 1:
            # Simple question — use regular retrieval
            return await self.aretrieve(
                question, top_k, user_filters, use_hyde, memory_context
            )

        # Parallel retrieval for each sub-question
        per_query_k = max(3, top_k // len(sub_questions) + 1)
        tasks = [
            self.aretrieve(sq, per_query_k, user_filters, use_hyde, memory_context)
            for sq in sub_questions
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge and deduplicate
        seen_content = set()
        merged = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Sub-query retrieval failed: {result}")
                continue
            for doc in result:
                # Use first 200 chars as dedup key
                content_key = doc.page_content[:200]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    merged.append(doc)

        logger.info(
            f"Decomposed retrieval: {len(sub_questions)} queries → {len(merged)} unique docs"
        )
        return merged[:top_k]

    # ── Re-ranking ────────────────────────────────────────────────────────

    async def arerank_documents(
        self,
        question: str,
        docs: List[Document],
        top_k: int = 6,
    ) -> List[Document]:
        """Re-rank documents using LLM-based relevance scoring.

        Applies after initial retrieval to improve precision.
        Uses a single LLM call to score all documents 0-10.

        Args:
            question: The user's question
            docs: Documents from initial retrieval
            top_k: Number of top documents to return

        Returns:
            Re-ranked documents (top_k most relevant)
        """
        if len(docs) <= top_k:
            return docs

        # Build document snippets for scoring
        docs_text = "\n\n".join([
            f"[DOC_{i}]\n{doc.page_content[:400]}"
            for i, doc in enumerate(docs)
        ])

        prompt = f"""Оцени релевантность каждого документа вопросу по шкале 0-10.
10 = идеально отвечает на вопрос
5 = частично релевантен
0 = совсем не относится к вопросу

ВОПРОС: {question}

ДОКУМЕНТЫ:
{docs_text}

Ответь в JSON формате:
{{"scores": [score_для_DOC_0, score_для_DOC_1, ...]}}

Массив должен содержать ровно {len(docs)} оценок."""

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
            data = json.loads(response.text or "{}")
            scores = data.get("scores", [])

            if len(scores) != len(docs):
                logger.warning(
                    f"Re-ranking returned {len(scores)} scores for {len(docs)} docs, skipping"
                )
                return docs[:top_k]

            # Sort by score descending
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked = [doc for doc, _ in scored_docs[:top_k]]
            logger.info(
                f"Re-ranked {len(docs)} docs → top {top_k}, "
                f"score range: {min(scores):.1f}-{max(scores):.1f}"
            )
            return reranked

        except Exception as e:
            logger.warning(f"Re-ranking failed, using original order: {e}")
            return docs[:top_k]

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

    # ── Tool/Function Calling ─────────────────────────────────────────────

    async def acheck_tool_need(
        self, question: str, memory_context: str = ""
    ) -> dict | None:
        """Check if the question requires a tool call.

        Returns:
            Dictionary with tool_name and arguments if tool needed,
            None otherwise.
        """
        from crag.tools import get_tool_schemas

        tool_schemas = get_tool_schemas()
        tools_desc = json.dumps(tool_schemas, ensure_ascii=False, indent=2)

        prompt = f"""Проанализируй вопрос пользователя и определи, нужен ли инструмент.

ВОПРОС: {question}

КОНТЕКСТ ПОЛЬЗОВАТЕЛЯ: {memory_context[:500] if memory_context else 'Нет'}

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools_desc}

ПРАВИЛА:
1. Используй check_deadline ТОЛЬКО если спрашивают о конкретном дедлайне для конкретного вуза
2. Используй calculate_budget ТОЛЬКО если спрашивают о расходах/бюджете в конкретном городе
3. Используй get_document_checklist ТОЛЬКО если просят чек-лист документов

Если вопрос общий или теоретический — НЕ используй инструменты.

Ответь СТРОГО в JSON формате:
{{"use_tool": true/false, "tool_name": "...", "arguments": {{...}}}}

Если инструмент не нужен:
{{"use_tool": false}}"""

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
            data = json.loads(response.text or "{}")

            if data.get("use_tool"):
                logger.info(
                    f"Tool needed: {data.get('tool_name')} "
                    f"with args {data.get('arguments')}"
                )
                return {
                    "tool_name": data.get("tool_name"),
                    "arguments": data.get("arguments", {}),
                }
        except Exception as e:
            logger.warning(f"Tool check failed: {e}")

        return None

    async def aexecute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool and return the result."""
        from crag.tools import execute_tool
        return await execute_tool(tool_name, arguments)

    async def astream_answer_with_tools(
        self,
        question: str,
        context: str,
        chat_history: str,
        memory_context: str,
        current_date: str,
        tool_result: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream answer, optionally incorporating tool results.

        If tool_result is provided, it's added to the context for the LLM.
        """
        # Enhance context with tool result if available
        enhanced_context = context
        if tool_result:
            tool_text = json.dumps(tool_result, ensure_ascii=False, indent=2)
            enhanced_context = (
                f"[РЕЗУЛЬТАТ ИНСТРУМЕНТА — используй эти ТОЧНЫЕ данные в ответе]\n"
                f"{tool_text}\n\n"
                f"[ДОКУМЕНТЫ ИЗ БАЗЫ ЗНАНИЙ]\n{context}"
            )

        # Use the regular streaming method with enhanced context
        async for chunk in self.astream_answer(
            question, enhanced_context, chat_history, memory_context, current_date
        ):
            yield chunk

