# Re-indexing Knowledge Base

## When to Re-index

You need to re-index the knowledge base when:
- ✅ Changing embedding provider (Google → NVIDIA, NVIDIA → OpenAI, etc.)
- ✅ Changing embedding model (different dimensions)
- ✅ Adding/modifying/deleting facts in `facts/` (universities, language, financial)
- ✅ Fixing dimension mismatch errors

## Current Configuration

The indexing script (`init_scripts/index_knowledge_base.py`) now:
- 🔄 Automatically detects embedding dimensions from configured provider
- 🔍 Verifies actual DB vector dimensions from stored vectors (`vector_dims`) to catch silent mismatches
- 🔄 Recreates database table if dimensions changed
- 🔄 Supports all providers (Google, NVIDIA, OpenAI)
- 🔄 Tracks provider and dimensions in `kb_sync_state` table

## How to Re-index (Docker Compose)

### Method 1: Force re-index on container restart

```bash
# In .env (or as an environment variable before docker compose up)
export FORCE_REINDEX=true

docker compose up -d --build
```

The init script runs automatically on every container start via `init_scripts/entry.sh`:
```bash
python3 init_scripts/init_bot_db.py
python3 -m bot.migrate
python3 init_scripts/index_knowledge_base.py
python3 bot/app.py
```

After a successful re-index, remove `FORCE_REINDEX` from `.env` to avoid recomputing embeddings on every restart.

### Method 2: Run the script manually inside the running container

```bash
docker compose exec freshmanragbot python3 -m init_scripts.index_knowledge_base
```

### Method 3: Local re-index (without Docker)

```bash
# Set up environment variables
export LLM_PROVIDER=nvidia
export NVIDIA_API_KEY=nvapi-...
export EMBEDDING_PROVIDER=nvidia
export EMBEDDING_MODEL=nvidia/nv-embed-v1
export POSTGRES_USER=...
export POSTGRES_PASSWORD=...
export POSTGRES_HOST=...
export POSTGRES_DB=...

# Run indexing
cd /path/to/project
python3 -m init_scripts.index_knowledge_base
```

## Verification

Check the bot container logs after re-indexing:

```bash
docker compose logs -f freshmanragbot
```

Expected output:
```
✅ Embedding provider: nvidia, dimensions: 1536
🔄 Embedding provider changed: google → nvidia
🔄 Embedding dimensions changed: 3072 → 1536
Recreating simple_documents table with new embedding dimensions...
Loaded 450 chunks total. Generating embeddings with nvidia...
Successfully indexed knowledge base. 450 chunks added.
```

## Fixing Dimension Mismatch Error

If you see error like:
```
asyncpg.exceptions.DataError: different vector dimensions 3072 and 1536
```

This means the database has old embeddings with different dimensions. Re-indexing will fix it automatically by:
1. Detecting old dimension (3072) vs new dimension (1536)
2. Dropping and recreating `simple_documents` table
3. Generating all embeddings with the new provider

Set `FORCE_REINDEX=true` and restart the bot container (`docker compose up -d --build`) and the table will be recreated automatically.

## Performance Notes

- **Small KB (<500 chunks):** Indexing takes ~2-3 minutes
- **Rate limits:** Script sleeps 2s between batches (100 chunks each) to avoid rate limits
- **API costs:**
  - Google Gemini: Free (up to 1500 requests/day)
  - NVIDIA: Free (rate-limited)
  - OpenAI: Paid (~$0.0001 per 1K tokens)

## Skipping Re-index

The script automatically skips re-indexing if:
- ✅ Knowledge base files haven't changed (MD5 hash match)
- ✅ Embedding provider hasn't changed
- ✅ Embedding dimensions haven't changed
- ✅ Database has data

This saves API quota on every container restart when the KB is unchanged.
