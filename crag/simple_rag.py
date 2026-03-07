import os
import json
import logging
from dataclasses import dataclass
from typing import List, AsyncGenerator
from sqlalchemy import text
from google import genai
from google.genai import types
import asyncio

logger = logging.getLogger(__name__)

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

class SimpleRAG:
    """A direct implementation of RAG without LangChain/LangGraph overhead."""
    
    def __init__(self, db_engine, prompts_config: dict = None):
        self.api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY"))
        self.client = genai.Client()
        self.db_engine = db_engine
        
        self.rag_prompt_messages = []
        self.system_prompt = ""
        self.human_prompt_template = "{question}\nКонтекст:\n{context}"
        self.rewrite_prompt_template = ""
        self.grading_prompt_template = ""
        
        if prompts_config:
            self.rag_prompt_messages = prompts_config.get("rag_prompt", {}).get("messages", [])
            self.system_prompt = self._extract_system_prompt(self.rag_prompt_messages)
            self.human_prompt_template = self._extract_human_prompt(self.rag_prompt_messages)
            
            # Load rewriting prompt template
            rewriting_messages = prompts_config.get("rewriting_prompt", {}).get("messages", [])
            self.rewrite_prompt_template = self._extract_human_prompt_from_messages(rewriting_messages)
            self.rewrite_system_prompt = self._extract_system_prompt(rewriting_messages)
            
            # Load grading prompt template
            grading_messages = prompts_config.get("grading_prompt", {}).get("messages", [])
            self.grading_prompt_template = self._extract_human_prompt_from_messages(grading_messages)
            self.grading_system_prompt = self._extract_system_prompt(grading_messages)

    def _extract_system_prompt(self, messages: list) -> str:
        for m in messages:
            if m.get("_target_") == "langchain_core.messages.SystemMessage":
                return m.get("content", "")
        return "Ты — полезный ассистент."

    def _extract_human_prompt(self, messages: list) -> str:
        for m in messages:
            if m.get("_target_") == "langchain_core.prompts.HumanMessagePromptTemplate.from_template":
                return m.get("template", "")
        return "Вопрос: {question}\nКонтекст: {context}\nИстория: {chat_history}\nПамять: {memory_context}"

    def _extract_human_prompt_from_messages(self, messages: list) -> str:
        """Same as above, alias for clarity."""
        return self._extract_human_prompt(messages)

    def get_embedding(self, text_input: str) -> List[float]:
        if not text_input.strip():
            return []
        try:
            response = self.client.models.embed_content(
                model='models/gemini-embedding-001',
                contents=text_input,
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []

    @staticmethod
    def _build_filter_sql(user_filters: dict) -> tuple[str, dict]:
        """Build SQL WHERE clause and ORDER BY hint from user filters.
        
        - Filters by country_scope (always includes 'ALL').
        - Soft-boosts results matching user's level via ORDER BY priority.
        
        Returns (where_clause_str, params_dict, level_boost_expr).
        """
        conditions = []
        params = {}
        
        country = user_filters.get("country_scope")
        if country and country != "ALL":
            conditions.append(
                "(metadata->'country_scope' @> '\"ALL\"'::jsonb "
                "OR metadata->'country_scope' @> (:country_filter)::jsonb)"
            )
            params["country_filter"] = json.dumps(country)  # e.g. '"RU"'
        
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        return where_clause, params

    def _build_level_boost_expr(self, user_filters: dict) -> tuple[str, dict]:
        """Build a level soft-boost expression for ORDER BY.
        
        Returns docs matching the user's level first, then everything else.
        """
        level = user_filters.get("level")
        if not level:
            return "", {}
        return (
            "CASE WHEN metadata->'level' @> (:user_level)::jsonb THEN 0 ELSE 1 END",
            {"user_level": json.dumps([level])}
        )

    def _db_retrieve(self, sql_str: str, all_params: dict) -> List[Document]:
        """Synchronous DB query — called via run_in_executor to avoid blocking event loop."""
        docs = []
        with self.db_engine.connect() as conn:
            result = conn.execute(text(sql_str), all_params)
            for row in result:
                docs.append(Document(page_content=row[0], metadata=row[1]))
        return docs

    async def aretrieve(self, query: str, top_k: int = 6, user_filters: dict = None) -> List[Document]:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.get_embedding, query)
        
        if not embedding:
            return []

        # Build optional metadata filter
        where_clause = ""
        filter_params = {}
        level_boost_expr = ""
        level_params = {}

        if user_filters:
            where_clause, filter_params = self._build_filter_sql(user_filters)
            level_boost_expr, level_params = self._build_level_boost_expr(user_filters)

        # Build ORDER BY: level boost first, then cosine similarity
        if level_boost_expr:
            order_by = f"ORDER BY {level_boost_expr} ASC, embedding <=> CAST(:embedding AS vector)"
        else:
            order_by = "ORDER BY embedding <=> CAST(:embedding AS vector)"

        sql_str = f"""
            SELECT content, metadata
            FROM simple_documents
            {where_clause}
            {order_by}
            LIMIT :top_k
        """
        
        all_params = {
            "embedding": str(embedding),
            "top_k": top_k,
            **filter_params,
            **level_params,
        }
        
        # Run blocking DB call in thread pool to not block event loop
        docs = await loop.run_in_executor(None, self._db_retrieve, sql_str, all_params)
        return docs

    async def arewrite_query(self, question: str, chat_history: str, memory_context: str) -> str:
        """Rewrite user question for better vector search using conversation context.
        
        Resolves anaphora (e.g. 'расскажи подробнее', 'а там?'), enriches with
        domain keywords, and clarifies ambiguous references.
        Gracefully falls back to the original question on any error.
        """
        if not self.rewrite_prompt_template:
            return question

        user_prompt = (
            self.rewrite_prompt_template
            .replace("{question}", question)
            .replace("{chat_history}", chat_history)
            .replace("{memory_context}", memory_context)
        )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, lambda: self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.rewrite_system_prompt,
                    temperature=0.0,
                )
            ))
            rewritten = (response.text or "").strip()
            if rewritten:
                logger.info(f"Query rewrite: '{question[:60]}' → '{rewritten[:60]}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")
        return question

    async def _grade_one(self, question: str, doc: Document) -> int:
        """Grade a single document for relevance to the question. Returns 0 or 1."""
        user_prompt = (
            self.grading_prompt_template
            .replace("{question}", question)
            .replace("{document}", doc.page_content[:800])
        )
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, lambda: self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.grading_system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            ))
            raw = (response.text or "{}").strip()
            data = json.loads(raw)
            return int(data.get("score", 1))
        except Exception as e:
            logger.warning(f"Grading failed for doc, keeping it: {e}")
            return 1  # fail-open: keep document if grading errors

    async def agrade_documents(self, question: str, docs: List[Document]) -> List[Document]:
        """Filter docs by relevance using CRAG grading prompt.
        
        Runs all grading calls concurrently. Falls back to full doc list
        if all docs get filtered (prevents empty-context hallucination prompt).
        """
        if not self.grading_prompt_template or not docs:
            return docs

        scores = await asyncio.gather(
            *[self._grade_one(question, doc) for doc in docs],
            return_exceptions=False,
        )
        relevant = [doc for doc, score in zip(docs, scores) if score == 1]
        
        if relevant:
            logger.info(f"Grading: {len(docs)} → {len(relevant)} relevant docs")
            return relevant
        else:
            # All filtered — fall back to original set to avoid empty context
            logger.info(f"Grading filtered all docs, falling back to full set ({len(docs)})")
            return docs

    async def astream_answer(self, question: str, context: str, chat_history: str, memory_context: str, current_date: str) -> AsyncGenerator[str, None]:
        sys_prompt = self.system_prompt.replace("{current_date}", current_date)
        user_prompt = self.human_prompt_template.replace("{question}", question) \
                                                .replace("{memory_context}", memory_context) \
                                                .replace("{chat_history}", chat_history) \
                                                .replace("{context}", context)

        # Define schema for structured output to guarantee format
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "answer": {"type": "STRING", "description": "The main text of the response to the user question."},
                "suggested_questions": {
                    "type": "ARRAY", 
                    "items": {"type": "STRING"},
                    "description": "3-4 suggested follow-up questions relevant to the current state."
                }
            },
            "required": ["answer", "suggested_questions"]
        }

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, lambda: self.client.models.generate_content_stream(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=response_schema
                )
            ))
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Error streaming from Gemini: {e}")
            yield json.dumps({
                "answer": "Извините, произошла ошибка при генерации ответа.",
                "suggested_questions": []
            })

    async def aadd_documents(self, docs: List[Document]) -> List[int]:
        """Add new documents, chunking and embedding manually."""
        loop = asyncio.get_event_loop()
        ids = []
        for doc in docs:
            embedding = await loop.run_in_executor(None, self.get_embedding, doc.page_content)
            if not embedding:
                continue
            with self.db_engine.begin() as conn:
                res = conn.execute(
                    text("INSERT INTO simple_documents (content, metadata, embedding) VALUES (:c, :m, :e) RETURNING id"), 
                    {"c": doc.page_content, "m": json.dumps(doc.metadata), "e": str(embedding)}
                )
                ids.append(res.scalar())
        return ids

    async def adelete_documents(self, ids: List[str]) -> bool:
        """Delete by id from DB."""
        try:
            with self.db_engine.begin() as conn:
                for doc_id in ids:
                    conn.execute(text("DELETE FROM simple_documents WHERE id = :id"), {"id": int(doc_id)})
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def agenerate_json(self, prompt: str) -> str:
        """Helper for memory updating"""
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, lambda: self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            ))
            return response.text
        except Exception as e:
            logger.error(f"Error generating JSON: {e}")
            return "{}"
