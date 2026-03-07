import asyncio
import os
import re
import yaml
import logging
import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field

from google import genai
from google.genai import types

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
    confidence: int = Field(description="Confidence score in the assessment from 1 to 100")
    reasoning: str = Field(description="Brief explanation of why this verdict was reached")


system_instruction = """
Ты — старший эксперт-аудитор австрийских образовательных программ.
Твоя задача — проверить, не устарела ли информация в нашей базе данных (Markdown файл) 
по сравнению с актуальным текстом, который мы только что скачали с официального сайта университета (Website Text).

ИНСТРУКЦИЯ:
1. Внимательно прочитай локальный Markdown-файл.
2. Прочитай сырой текст с веб-сайта.
3. Сравни факты, обращая особое внимание на:
   - Дедлайны (сроки подачи документов)
   - Цены и пошлины (ÖH-Beitrag, Tuition Fees)
   - Требования к языку (какие сертификаты нужны, какой уровень)
   - Списки вступительных экзаменов или процедур отбора
4. ЕСЛИ на сайте есть новая, противоречащая информация — статус NEEDS_UPDATE.
5. ЕСЛИ сайт содержит меньше информации, чем Markdown (например, специфичные детали убрали в PDF), но нет прямых противоречий — статус UP_TO_DATE.
6. ЕСЛИ всё совпадает (или сайт просто подтверждает наши данные) — статус UP_TO_DATE.
"""

async def fetch_url_text(url: str, session: aiohttp.ClientSession) -> str:
    """Fetches URL and extracts visible text."""
    try:
        # Some university sites block generic bots
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
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

        # Extract frontmatter
        match = re.search(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
        if not match:
            return file_path, VerificationResult(
                status="ERROR", 
                changes_detected=["No YAML frontmatter found"], 
                confidence=0, 
                reasoning="Invalid file format"
            )

        frontmatter_str = match.group(1)
        markdown_body = match.group(2).strip()

        try:
            metadata = yaml.safe_load(frontmatter_str)
        except Exception as e:
            return file_path, VerificationResult(status="ERROR", changes_detected=[str(e)], confidence=0, reasoning="YAML parse error")

        source_url_raw = metadata.get("source_url", "")
        if not source_url_raw or "example.at" in source_url_raw:
            return file_path, VerificationResult(
                status="UP_TO_DATE", 
                changes_detected=[], 
                confidence=100, 
                reasoning="No valid source_url to check"
            )

        # Handle multiple URLs separated by |
        urls = [u.strip() for u in source_url_raw.split("|")]
        if not urls:
            return file_path, VerificationResult(status="ERROR", changes_detected=[], confidence=0, reasoning="Empty source_url")

        # Fetch all URLs
        logger.info(f"[{os.path.basename(file_path)}] Fetching {len(urls)} URLs...")
        live_texts = []
        for u in urls:
            # remove inline comments
            u = u.split('#')[0].strip()
            text = await fetch_url_text(u, session)
            live_texts.append(f"--- SOURCE: {u} ---\n{text}\n")
            
        full_live_text = "\n".join(live_texts)

        # Call Gemini
        logger.info(f"[{os.path.basename(file_path)}] Asking Gemini to verify...")
        
        prompt = f"""
        LOCAL MARKDOWN DOCUMENT:
        ========================
        {markdown_body}
        
        LIVE WEBSITE TEXT:
        ==================
        {full_live_text}
        """
        
        try:
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=VerificationResult,
                    temperature=0.1 # Low temp for factual consistency
                )
            )
            result = VerificationResult.model_validate_json(response.text)
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
    for root, _, files in os.walk(kb_dir):
        for f in files:
            if f.endswith(".md") and not f.startswith("_"):
                if f in ("README.md", "schema.yaml"):
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
        if result.changes_detected:
            print("   Detected Changes:")
            for change in result.changes_detected:
                print(f"     - {change}")
                
        # Markdown File output
        report_lines.append(f"## {icon} {fname} `{result.status}` (Confidence: {result.confidence}%)")
        report_lines.append(f"**Reasoning:** {result.reasoning}\n")
        if result.changes_detected:
            report_lines.append("**Detected Changes:**")
            for change in result.changes_detected:
                report_lines.append(f"- {change}")
        report_lines.append("---\n")
                
    with open("kb_audit_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    logger.info(f"Full report saved to kb_audit_report.md")

if __name__ == "__main__":
    asyncio.run(main())
