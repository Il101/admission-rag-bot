import asyncio
import logging
import os
import re
import hashlib
import yaml
import json
import sys
from pathlib import Path
from sqlalchemy import create_engine, text

# Add parent directory to path to import crag module
sys.path.insert(0, str(Path(__file__).parent.parent))

from crag.llm_providers import get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── YAML Frontmatter Parsing ──────────────────────────────────────────────

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from a markdown file.
    
    Returns (metadata_dict, body_text). If no frontmatter found,
    returns ({}, full_content).
    """
    if not content.startswith("---"):
        return {}, content

    end = content.find("---", 3)
    if end == -1:
        return {}, content

    frontmatter_str = content[3:end].strip()
    body = content[end + 3:].strip()

    try:
        metadata = yaml.safe_load(frontmatter_str)
        if not isinstance(metadata, dict):
            return {}, content
        return metadata, body
    except yaml.YAMLError:
        logger.warning(f"Failed to parse YAML frontmatter")
        return {}, content


# ── Markdown-Aware Chunking ───────────────────────────────────────────────

def split_markdown(body: str, file_metadata: dict, max_chunk_len: int = 800) -> list[dict]:
    """Split markdown body into chunks, preserving heading hierarchy.

    Each chunk inherits the chain of headings above it as a section_path,
    and gets a contextual prefix prepended for better embedding quality.
    
    Supports section-level URLs via <!-- url: https://... --> HTML comments.
    Each chunk gets the most specific URL available (section > parent > frontmatter).
    
    Returns list of {"content": str, "metadata": dict}.
    """
    lines = body.split("\n")
    
    # Track current heading hierarchy: {level: heading_text}
    heading_stack: dict[int, str] = {}
    # Track section-level URLs: {level: url}
    url_stack: dict[int, str] = {}
    
    # Collect sections: each section is (heading_stack_snapshot, url_snapshot, text_lines)
    sections: list[tuple[dict, dict, list[str]]] = []
    current_lines: list[str] = []
    current_headings: dict[int, str] = {}
    current_urls: dict[int, str] = {}
    current_heading_level: int = 0
    
    url_comment_re = re.compile(r'<!--\s*url:\s*(https?://\S+)\s*-->')
    
    for line in lines:
        # Check for <!-- url: ... --> comment
        url_match = url_comment_re.match(line.strip())
        if url_match:
            url = url_match.group(1)
            url_stack[current_heading_level] = url
            current_urls = dict(url_stack)
            continue  # Don't include the comment line in chunk content
        
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            # Save the current section before starting a new one
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append((dict(current_headings), dict(current_urls), current_lines[:]))
                current_lines = []
            
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            current_heading_level = level
            
            # Update heading stack: set this level and clear deeper levels
            heading_stack[level] = heading_text
            keys_to_remove = [k for k in heading_stack if k > level]
            for k in keys_to_remove:
                del heading_stack[k]
            
            # Clear deeper URL overrides too (new section resets child URLs)
            url_keys_to_remove = [k for k in url_stack if k > level]
            for k in url_keys_to_remove:
                del url_stack[k]
            
            # Snapshot current headings and URLs for this section
            current_headings = dict(heading_stack)
            current_urls = dict(url_stack)
            current_lines.append(line)
        else:
            current_lines.append(line)
    
    # Don't forget the last section
    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((dict(current_headings), dict(current_urls), current_lines[:]))
    
    # Build the document title from the H1 or from metadata
    doc_title = file_metadata.get("title", "")
    if not doc_title:
        # Try to get it from heading_stack level 1
        for headings, _, _ in sections:
            if 1 in headings:
                doc_title = headings[1]
                break
    if not doc_title:
        doc_title = file_metadata.get("source", "Документ").replace(".md", "").replace("-", " ").title()

    # Fallback URL from frontmatter (may contain multiple pipe-separated URLs; take first)
    fallback_url = file_metadata.get("source_url", "")
    if isinstance(fallback_url, str) and "|" in fallback_url:
        fallback_url = fallback_url.split("|")[0].strip()

    # Now build chunks from sections
    chunks = []
    
    for headings, urls, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if not section_text or len(section_text) < 30:
            continue
        
        # Build section_path from heading hierarchy (skip H1 which is the doc title)
        path_parts = []
        for lvl in sorted(headings.keys()):
            if lvl == 1:
                continue  # H1 is the doc title, already in prefix
            path_parts.append(headings[lvl])
        section_path = " > ".join(path_parts) if path_parts else ""
        
        # Resolve the best URL: deepest level in url_stack, or fallback
        section_url = fallback_url
        if urls:
            deepest_level = max(urls.keys())
            section_url = urls[deepest_level]
        
        # Build contextual prefix
        prefix_parts = []
        university = file_metadata.get("university", "")
        if university:
            prefix_parts.append(f"Университет: {university}")
        elif doc_title:
            prefix_parts.append(doc_title)
        
        topic = file_metadata.get("topic", "")
        if topic:
            prefix_parts.append(f"Тема: {topic}")
        
        if section_path:
            prefix_parts.append(f"Раздел: {section_path}")
        
        prefix = "[" + " | ".join(prefix_parts) + "]" if prefix_parts else ""
        
        # Build chunk metadata
        chunk_meta = {
            "source": file_metadata.get("source", ""),
            "title": doc_title,
            "section_path": section_path,
            "source_url": section_url,
        }
        # Copy other important fields from frontmatter
        for key in ("university", "topic", "country_scope", "level", "city", "related_docs"):
            if key in file_metadata:
                chunk_meta[key] = file_metadata[key]
        
        # If section is small enough, emit as a single chunk
        if len(section_text) <= max_chunk_len:
            content = f"{prefix}\n{section_text}" if prefix else section_text
            chunks.append({"content": content, "metadata": chunk_meta})
        else:
            # Split large sections by paragraphs with overlap
            sub_chunks = _split_long_section(section_text, max_chunk_len, overlap=50)
            for i, sub in enumerate(sub_chunks):
                sub_path = f"{section_path} (часть {i+1})" if section_path else f"часть {i+1}"
                sub_meta = dict(chunk_meta)
                sub_meta["section_path"] = sub_path
                content = f"{prefix}\n{sub}" if prefix else sub
                chunks.append({"content": content, "metadata": sub_meta})
    
    return chunks


def _split_long_section(text: str, max_length: int = 800, overlap: int = 50) -> list[str]:
    """Split a long text section by paragraphs, with character overlap."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: keep last N characters as start of next chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + p
            else:
                current_chunk = p
        else:
            current_chunk += "\n\n" + p if current_chunk else p
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Further break down giant paragraphs
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


# ── Main Indexing Logic ───────────────────────────────────────────────────

async def index():
    logger.info("Initializing indexing with multi-provider support...")

    # Initialize LLM provider to get embedding dimensions
    logger.info("Detecting embedding provider configuration...")
    llm_provider = get_llm()

    # Detect embedding dimensions by generating a test embedding
    test_embedding = await llm_provider.embed(["test"])
    embedding_dim = len(test_embedding[0])
    logger.info(f"✅ Embedding provider: {llm_provider.config.embedding_provider}, dimensions: {embedding_dim}")

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

        # Create or migrate kb_sync_state table
        # First, check if table exists with old schema
        try:
            result = conn.execute(text("SELECT * FROM kb_sync_state WHERE id = 1 LIMIT 1")).fetchone()
            # Try to read from old schema (only hash column)
            old_hash = result[0] if result else None
            old_dim = None
            old_provider = None
            # Table exists, needs migration
            logger.info("🔄 Migrating kb_sync_state table schema...")
            conn.execute(text("""
                ALTER TABLE kb_sync_state
                ADD COLUMN IF NOT EXISTS embedding_dim INT,
                ADD COLUMN IF NOT EXISTS provider TEXT
            """))
        except Exception:
            # Table doesn't exist, create new one
            conn.execute(text("CREATE TABLE IF NOT EXISTS kb_sync_state (id INT PRIMARY KEY, hash TEXT, embedding_dim INT, provider TEXT)"))
            result = conn.execute(text("SELECT hash, embedding_dim, provider FROM kb_sync_state WHERE id = 1")).fetchone()
            old_hash = result[0] if result else None
            old_dim = result[1] if result else None
            old_provider = result[2] if result else None

        # If we got here from migration, read the full state now
        if old_provider is None:
            result = conn.execute(text("SELECT hash, embedding_dim, provider FROM kb_sync_state WHERE id = 1")).fetchone()
            if result:
                old_hash, old_dim, old_provider = result
            else:
                old_hash, old_dim, old_provider = None, None, None

        # Check if embedding provider or dimensions changed
        provider_changed = old_provider != llm_provider.config.embedding_provider
        dimension_changed = old_dim != embedding_dim

        if provider_changed:
            logger.info(f"🔄 Embedding provider changed: {old_provider} → {llm_provider.config.embedding_provider}")
        if dimension_changed:
            logger.info(f"🔄 Embedding dimensions changed: {old_dim} → {embedding_dim}")

        # Drop and recreate table if dimensions changed
        if dimension_changed or provider_changed:
            logger.info("Recreating simple_documents table with new embedding dimensions...")
            conn.execute(text("DROP TABLE IF EXISTS simple_documents CASCADE"))

        # Ensure our target table exists with correct dimensions
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS simple_documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector({embedding_dim})
            )
        """))

        # GIN index for JSONB metadata filtering
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_simple_documents_metadata "
            "ON simple_documents USING GIN (metadata)"
        ))

        # Drop legacy HNSW index (pgvector 2000-dim limit makes it unusable for high-d embeddings)
        conn.execute(text(
            "DROP INDEX IF EXISTS idx_simple_documents_embedding_hnsw"
        ))

        # Full-text search index for hybrid retrieval (BM25-like)
        # Combined simple + russian config for multilingual KB (German/English + Russian)
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_simple_documents_content_fts "
            "ON simple_documents USING GIN ((to_tsvector('simple', content) || to_tsvector('russian', content)))"
        ))

        # Check if table has data. Even if hash matches, table might be empty
        try:
            count = conn.execute(text("SELECT count(*) FROM simple_documents")).scalar()
            # Check the actual embedding column vector dimension
            col_info = conn.execute(text("""
                SELECT data_type FROM information_schema.columns
                WHERE table_name='simple_documents' AND column_name='embedding'
            """)).scalar()

            # Parse vector dimension from data_type like "vector(3072)"
            if col_info and 'vector(' in col_info:
                actual_dim = int(col_info.split('(')[1].split(')')[0])
                if actual_dim != embedding_dim and count > 0:
                    logger.info(f"⚠️ Dimension mismatch: DB has {actual_dim}-dim vectors, provider has {embedding_dim}-dim")
                    logger.info("🔄 Forcing re-index to update embeddings...")
                    dimension_changed = True
        except Exception as e:
            logger.info(f"Could not check column info: {e}")
            count = 0

    # NOTE: pgvector HNSW/IVFFlat indexes are limited to 2000 dims, and
    # Railway's pgvector doesn't support halfvec.  For a small knowledge base
    # the exact sequential scan is fast enough (~0.4 s including network).

    if old_hash == current_hash and count > 0 and not provider_changed and not dimension_changed:
        logger.info("✅ Knowledge base is unchanged since last deployment. Skipping indexing to save API limits.")
        return

    logger.info("🔄 Knowledge base has changed or provider changed. Clearing existing indexes and re-indexing...")
    with engine.begin() as conn:
        # We drop the old langchain payload and our new simple tables
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS docstore CASCADE"))
        conn.execute(text("TRUNCATE TABLE simple_documents RESTART IDENTITY"))

    # Load and chunk markdown with metadata-aware splitting
    logger.info("Loading documents from knowledge_base/...")
    all_chunks = []
    for path in md_files:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        source = os.path.basename(path)
        
        # Skip non-content files (author docs, schema, registry)
        if source.startswith("_") or source in ("schema.yaml", "README.md"):
            continue
        
        # Parse YAML frontmatter and body
        frontmatter, body = parse_frontmatter(content)
        
        # Build file-level metadata
        file_metadata = {
            "source": source,
            **frontmatter,  # includes source_url, university, topic, country_scope, level, etc.
        }
        
        # Use markdown-aware chunking
        chunks = split_markdown(body, file_metadata, max_chunk_len=800)
        
        for chunk in chunks:
            if chunk["content"].strip():
                all_chunks.append(chunk)
        
        logger.info(f"  📄 {source}: {len(chunks)} chunks (frontmatter: {bool(frontmatter)})")

    if not all_chunks:
        logger.warning("No documents found in knowledge_base/!")
        return

    logger.info(f"Loaded {len(all_chunks)} chunks total. Generating embeddings with {llm_provider.config.embedding_provider}...")

    batch_size = 100
    total_added = 0

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        logger.info(f"Adding batch {i//batch_size + 1} ({len(batch)} chunks)...")

        texts_to_embed = [item["content"] for item in batch]

        # Use LLM provider for embeddings (supports Google, NVIDIA, OpenAI)
        embeddings = await llm_provider.embed(texts_to_embed)

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

    logger.info(f"Successfully indexed knowledge base. {total_added} chunks added.")

    # Save the new hash state with provider and dimension info
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO kb_sync_state (id, hash, embedding_dim, provider)
                VALUES (1, :hash, :dim, :provider)
                ON CONFLICT (id) DO UPDATE SET
                    hash = EXCLUDED.hash,
                    embedding_dim = EXCLUDED.embedding_dim,
                    provider = EXCLUDED.provider
            """),
            {
                "hash": current_hash,
                "dim": embedding_dim,
                "provider": llm_provider.config.embedding_provider
            }
        )

if __name__ == "__main__":
    asyncio.run(index())
