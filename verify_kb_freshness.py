import asyncio
import os
import re
import yaml
import logging
import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "init_scripts"))
from index_knowledge_base import parse_frontmatter, split_markdown


from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from google.genai.errors import APIError

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Verify API key
import getpass
raw_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not raw_key:
    raw_key = getpass.getpass("Enter your Gemini API key (GOOGLE_API_KEY): ")
    if not raw_key:
        logger.error("API key is required.")
        exit(1)

# Strip any surrounding quotes from .env file parsing
api_key = raw_key.strip("'\"")

client = genai.Client(api_key=api_key)

# Use Gemini 3 Flash Preview as requested
MODEL_NAME = "gemini-3-flash-preview"

# Structured output definition
class VerificationResult(BaseModel):
    status: str = Field(description="Must be exactly 'UP_TO_DATE', 'NEEDS_UPDATE', or 'ERROR'")
    changes_detected: list[str] = Field(description="List of factual differences found (e.g. deadline changed from X to Y). Empty if UP_TO_DATE.")
    additions_suggested: list[str] = Field(description="Valuable information present on the website but missing from the Markdown that would make it more informative.")
    deletions_suggested: list[str] = Field(description="Information present in the Markdown that is misleading, incorrect, or contradicts the website and should be removed.")
    confidence: int = Field(description="Confidence score in the assessment from 1 to 100")
    reasoning: str = Field(description="Brief explanation of why this verdict was reached")


system_instruction = """
Ты — старший эксперт-аудитор и редактор базы знаний по австрийским образовательным программам.
Твоя задача — не просто проверить базовую актуальность, но и глубоко проанализировать локальную базу данных (Markdown файл) по сравнению с текстом скачанным с официального сайта университета (Website Text).
Документ может быть разбит на несколько секций, каждая из которых имеет свой URL-источник на сайте университета.

ИНСТРУКЦИЯ:
1. Внимательно прочитай предоставленные секции локального Markdown-файла и соответствующие им тексты с веб-сайта.
2. Сравни факты, обращая особое внимание на дедлайны, цены, требования к языку и вступительные экзамены.
3. ОПРЕДЕЛЕНИЕ СТАТУСА:
   - ЕСЛИ найдена прямо противоречащая, устаревшая информация (например, изменился дедлайн) -> статус NEEDS_UPDATE. Заполняй `changes_detected`.
   - ЕСЛИ в Markdown есть информация, которая откровенно вводит в заблуждение или опровергается сайтом -> статус NEEDS_UPDATE. Заполняй `deletions_suggested`.
   - ЕСЛИ на сайте есть полезная, критически важная новая информация, которой не хватает в Markdown (и она сделает базу информативнее) -> статус NEEDS_UPDATE. Заполняй `additions_suggested`.
   - ЕСЛИ в Markdown информация просто более подробная (справочная), а сайт содержит краткую выжимку, НО нет прямых противоречий -> статус UP_TO_DATE.
   - ЕСЛИ всё совпадает идеально, либо сайт просто подтверждает наши данные -> статус UP_TO_DATE.
4. ЗАПОЛНЕНИЕ ПОЛЕЙ:
   - changes_detected: Только факты, которые изменились (было X, стало Y).
   - additions_suggested: Что стоит ДОБАВИТЬ в Markdown, так как это есть на сайте и важно для абитуриента.
   - deletions_suggested: Что стоит УДАЛИТЬ из Markdown, так как это неверно, вводит в заблуждение или больше не актуально по данным сайта.
"""

async def fetch_url_text(url: str, session: aiohttp.ClientSession) -> str:
    """Fetches URL and extracts visible text."""
    try:
        # Some university sites block generic bots
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate", # Prevent Brotli ('br') encoding which aiohttp struggles with natively
        }
        # Disable SSL verification for misconfigured university servers if needed
        async with session.get(url, headers=headers, ssl=False, timeout=15) as response:
            if response.status != 200:
                return f"HTTP ERROR: {response.status}"
            html = await response.text()
            
            # Clean HTML to just get text
            soup = BeautifulSoup(html, "html.parser")
            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "footer"]):
                script.extract()
            
            text = soup.get_text(separator="\n")
            # Collapse multiple newlines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)
    except Exception as e:
        return f"FETCH ERROR: {str(e)}"

async def verify_document(file_path: str, client: genai.Client, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
    """Processes a single markdown file."""
    async with semaphore:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        file_metadata, markdown_body = parse_frontmatter(content)
        if not file_metadata:
            return file_path, VerificationResult(
                status="ERROR", 
                changes_detected=["No YAML frontmatter found"], 
                confidence=0, 
                reasoning="Invalid file format"
            )

        file_metadata["source"] = os.path.basename(file_path)
        chunks = split_markdown(markdown_body, file_metadata)
        
        chunks_by_url = {}
        for chunk in chunks:
            url = chunk["metadata"].get("source_url", "")
            if url:
                url = url.split('#')[0].split('|')[0].strip() # Clean inline comments or multiple URLs
                if url not in chunks_by_url:
                    chunks_by_url[url] = []
                chunks_by_url[url].append(chunk)

        if not chunks_by_url:
            return file_path, VerificationResult(
                status="UP_TO_DATE", 
                changes_detected=[], 
                confidence=100, 
                reasoning="No valid source_url to check"
            )

        logger.info(f"[{os.path.basename(file_path)}] Fetching {len(chunks_by_url)} unique URLs...")
        url_sections = []
        for url, url_chunks in chunks_by_url.items():
            if not url or "example.at" in url:
                continue
            text = await fetch_url_text(url, session)
            local_text = "\n\n".join(c["content"] for c in url_chunks)
            
            section = f"--- URL: {url} ---\nLOCAL MARKDOWN CONTENT SPECIFIC TO THIS URL:\n{local_text}\n\nLIVE WEBSITE TEXT FROM THIS URL:\n{text}\n\n"
            url_sections.append(section)

        if not url_sections:
            return file_path, VerificationResult(
                status="UP_TO_DATE", 
                changes_detected=[], 
                confidence=100, 
                reasoning="No valid URLs to check"
            )

        # Call Gemini
        logger.info(f"[{os.path.basename(file_path)}] Asking Gemini to verify across {len(url_sections)} URLs...")
        
        prompt = f"""
        We have broken down the local document into sections based on their source URLs.
        Please compare the local content for each URL against the live website text for that specific URL.
        
        {''.join(url_sections)}
        """
        
        # Define a retryable inner function for the Gemini call
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=4, max=60),
            retry=retry_if_exception_type(APIError),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        async def call_gemini():
            res = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=VerificationResult,
                    temperature=0.1 # Low temp for factual consistency
                )
            )
            return VerificationResult.model_validate_json(res.text)

        try:
            result = await call_gemini()
            return file_path, result
        except Exception as e:
            return file_path, VerificationResult(
                status="ERROR", 
                changes_detected=[f"Gemini API Error: {str(e)}"], 
                confidence=0, 
                reasoning="LLM generation failed"
            )

async def main():
    kb_dir = "knowledge_base"
    if not os.path.exists(kb_dir):
        logger.error(f"Directory {kb_dir} not found.")
        return

    # Find all MD files
    md_files = []
    
    # Optional file filter from CLI
    import sys
    filter_files = sys.argv[1:]

    for root, _, files in os.walk(kb_dir):
        for f in files:
            if f.endswith(".md") and not f.startswith("_"):
                if f in ("README.md", "schema.yaml"):
                    continue
                if filter_files and not any(filter_name in f for filter_name in filter_files):
                    continue
                md_files.append(os.path.join(root, f))
                
    # Run on ALL relevant files
    test_files = md_files
    
    logger.info(f"Starting verification run for all {len(test_files)} files in the knowledge base...")
    
    # 2 concurrent files to respect API rate limits
    semaphore = asyncio.Semaphore(2) 
    
    async with aiohttp.ClientSession() as session:
        tasks = [verify_document(f, client, session, semaphore) for f in test_files]
        results = await asyncio.gather(*tasks)

    # Print Report & Write to File
    report_lines = []
    report_lines.append("# Knowledge Base Verification Report")
    report_lines.append("Generated by AI comparing local Markdown files against live university website data.\n")
    
    print("\n" + "="*60)
    print("🎓 KNOWLEDGE BASE VERIFICATION REPORT 🎓")
    print("="*60)
    
    for file_path, result in results:
        fname = os.path.basename(file_path)
        if result.status == "UP_TO_DATE":
            icon = "✅"
            color_start = "\033[92m" # Green
        elif result.status == "NEEDS_UPDATE":
            icon = "⚠️"
            color_start = "\033[93m" # Yellow
        else:
            icon = "❌"
            color_start = "\033[91m" # Red
        color_end = "\033[0m"
        
        # Console output
        print(f"\n{color_start}{icon} {fname} [{result.status}]{color_end} (Confidence: {result.confidence}%)")
        print(f"   Reasoning: {result.reasoning}")
        if result.changes_detected or result.additions_suggested or result.deletions_suggested:
            print("   Issues Found:")
            for change in result.changes_detected:
                print(f"     - [CHANGE] {change}")
            for add in result.additions_suggested:
                print(f"     - [ADD] {add}")
            for dele in result.deletions_suggested:
                print(f"     - [DELETE] {dele}")
                
        # Markdown File output
        report_lines.append(f"## {icon} {fname} `{result.status}` (Confidence: {result.confidence}%)")
        report_lines.append(f"**Reasoning:** {result.reasoning}\n")
        if result.changes_detected:
            report_lines.append("**Detected Changes:**")
            for change in result.changes_detected:
                report_lines.append(f"- [CHANGE] {change}")
        if result.additions_suggested:
            report_lines.append("**Suggested Additions:**")
            for add in result.additions_suggested:
                report_lines.append(f"- [ADD] {add}")
        if result.deletions_suggested:
            report_lines.append("**Suggested Deletions:**")
            for dele in result.deletions_suggested:
                report_lines.append(f"- [DELETE] {dele}")
        report_lines.append("---\n")
                
    with open("kb_audit_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    logger.info(f"Full report saved to kb_audit_report.md")

if __name__ == "__main__":
    asyncio.run(main())
