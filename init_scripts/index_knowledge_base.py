import asyncio
import logging
import os
import hashlib
import yaml
import json
from pathlib import Path
from sqlalchemy import create_engine, text
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_text(text_content: str, max_length: int = 500) -> list[str]:
    """A very simple recursive character splitter equivalent."""
    paragraphs = text_content.split('\n\n')
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = p
        else:
            current_chunk += "\n\n" + p if current_chunk else p
            
    if current_chunk:
        chunks.append(current_chunk.strip())
    # Further break down giant paragraphs if needed
    final_chunks = []
    for c in chunks:
        if len(c) > max_length * 2:
            sentences = c.replace('. ', '.\n').split('\n')
            temp = ""
            for s in sentences:
                if len(temp) + len(s) > max_length and temp:
                    final_chunks.append(temp.strip())
                    temp = s
                else:
                    temp += " " + s if temp else s
            if temp:
                final_chunks.append(temp.strip())
        else:
            final_chunks.append(c)
            
    return final_chunks

async def index():
    logger.info("Initializing pipeline without LangChain...")
    
    # Check knowledge base state
    logger.info("Checking knowledge base state...")
    kb_dir = "knowledge_base"
    md_files = []
    for root, dirs, files in os.walk(kb_dir):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    md_files.sort()
    
    hashes = []
    for path in md_files:
        with open(path, 'rb') as f:
            hashes.append(hashlib.md5(f.read()).hexdigest())
    current_hash = hashlib.md5("".join(hashes).encode()).hexdigest()

    # Get DB URL from yaml
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    db_url_template = config.get("bot_db_connection")
    db_url = db_url_template.replace("${oc.env:POSTGRES_USER}", os.environ.get("POSTGRES_USER", "")) \
                            .replace("${oc.env:POSTGRES_PASSWORD}", os.environ.get("POSTGRES_PASSWORD", "")) \
                            .replace("${oc.env:POSTGRES_HOST}", os.environ.get("POSTGRES_HOST", "")) \
                            .replace("${oc.env:POSTGRES_DB}", os.environ.get("POSTGRES_DB", ""))

    engine = create_engine(db_url)
    
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS kb_sync_state (id INT PRIMARY KEY, hash TEXT)"))
        result = conn.execute(text("SELECT hash FROM kb_sync_state WHERE id = 1")).fetchone()
        old_hash = result[0] if result else None

        # Ensure our target table exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS simple_documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector
            )
        """))
        
        # Check if table has data. Even if hash matches, table might be empty
        count = conn.execute(text("SELECT count(*) FROM simple_documents")).scalar()

    if old_hash == current_hash and count > 0:
        logger.info("✅ Knowledge base is unchanged since last deployment. Skipping indexing to save API limits.")
        return

    logger.info("🔄 Knowledge base has changed or is empty. Clearing existing indexes and re-indexing...")
    with engine.begin() as conn:
        # We drop the old langchain payload and our new simple tables
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS docstore CASCADE"))
        conn.execute(text("TRUNCATE TABLE simple_documents RESTART IDENTITY"))

    # Load and chunk markdown
    logger.info("Loading documents from knowledge_base/...")
    all_chunks = []
    for path in md_files:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Simple metadata extraction
            source = os.path.basename(path)
            title = source.replace(".md", "").replace("-", " ").title()
            chunks = split_text(content, max_length=500)
            for c in chunks:
                if c.strip():
                    all_chunks.append({
                        "content": c,
                        "metadata": {"source": source, "title": title}
                    })

    if not all_chunks:
        logger.warning("No documents found in knowledge_base/!")
        return
        
    logger.info(f"Loaded {len(all_chunks)} chunks. Generating embeddings...")
    client = genai.Client()
    
    batch_size = 100
    total_added = 0
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        logger.info(f"Adding batch {i//batch_size + 1} ({len(batch)} chunks)...")
        
        texts_to_embed = [item["content"] for item in batch]
        
        # Google GenAI lets you embed a list of strings
        response = client.models.embed_content(
            model='models/gemini-embedding-001',
            contents=texts_to_embed,
        )
        
        embeddings = [e.values for e in response.embeddings]
        
        # Insert into DB
        with engine.begin() as conn:
            for item, emb in zip(batch, embeddings):
                conn.execute(
                    text("INSERT INTO simple_documents (content, metadata, embedding) VALUES (:c, :m, :e)"), 
                    {"c": item["content"], "m": json.dumps(item["metadata"]), "e": str(emb)}
                )
        total_added += len(batch)
        
        if i + batch_size < len(all_chunks):
            logger.info("Sleeping 2s to avoid rate limits...")
            await asyncio.sleep(2)

    logger.info(f"Successfully indexed knowledge base. IDs: {total_added} chunks added.")
    
    # Save the new hash state
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO kb_sync_state (id, hash) VALUES (1, :hash) ON CONFLICT (id) DO UPDATE SET hash = EXCLUDED.hash"), 
            {"hash": current_hash}
        )

if __name__ == "__main__":
    asyncio.run(index())
