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
    return "\n\n".join(doc.page_content for doc in docs)

class SimpleRAG:
    """A direct implementation of RAG without LangChain/LangGraph overhead."""
    
    def __init__(self, db_engine, prompts_config: dict = None):
        self.api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY"))
        self.client = genai.Client()
        self.db_engine = db_engine
        
        self.rag_prompt_messages = []
        self.system_prompt = ""
        self.human_prompt_template = "{question}\nКонтекст:\n{context}"
        
        if prompts_config:
            self.rag_prompt_messages = prompts_config.get("rag_prompt", {}).get("messages", [])
            self.system_prompt = self._extract_system_prompt(self.rag_prompt_messages)
            self.human_prompt_template = self._extract_human_prompt(self.rag_prompt_messages)

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

    async def aretrieve(self, query: str, top_k: int = 4) -> List[Document]:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.get_embedding, query)
        
        if not embedding:
            return []

        sql = text("""
            SELECT content, metadata
            FROM simple_documents
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)
        
        docs = []
        # Uses standard psycopg SQLAlchemy synchronous execution locally
        with self.db_engine.connect() as conn:
            result = conn.execute(sql, {"embedding": str(embedding), "top_k": top_k})
            for row in result:
                docs.append(Document(page_content=row[0], metadata=row[1]))
                
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
                # ids is a list of strings but DB id is INT
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
