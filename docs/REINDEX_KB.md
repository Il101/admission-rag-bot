# Re-indexing Knowledge Base

## When to Re-index

You need to re-index the knowledge base when:
- ✅ Changing embedding provider (Google → NVIDIA, NVIDIA → OpenAI, etc.)
- ✅ Changing embedding model (different dimensions)
- ✅ Adding/modifying/deleting documents in `knowledge_base/`
- ✅ Fixing dimension mismatch errors

## Current Configuration

The indexing script (`init_scripts/index_knowledge_base.py`) now:
- 🔄 Automatically detects embedding dimensions from configured provider
- 🔍 Verifies actual DB vector dimensions from stored vectors (`vector_dims`) to catch silent mismatches
- 🔄 Recreates database table if dimensions changed
- 🔄 Supports all providers (Google, NVIDIA, OpenAI)
- 🔄 Tracks provider and dimensions in `kb_sync_state` table

## How to Re-index on Railway

### Method 1: Trigger Re-deployment

The simplest way is to trigger a re-deployment, which runs the indexing script automatically:

```bash
# Change any environment variable to trigger redeployment
railway variables set REINDEX_TRIGGER=$(date +%s)

# Or just redeploy
railway up --detach
```

If you need to force a full rebuild even when hash/provider look unchanged:

```bash
railway variable set FORCE_REINDEX=true
railway up --detach
```

After successful re-index, disable it to avoid unnecessary work on every deploy:

```bash
railway variable delete FORCE_REINDEX
```

The init script runs automatically on every deployment via container start command:
```bash
sh init_scripts/entry.sh
```

Actual startup sequence in `init_scripts/entry.sh`:
```bash
python3 init_scripts/init_bot_db.py
python3 -m bot.migrate
python3 init_scripts/index_knowledge_base.py
python3 bot/app.py
```

### Method 2: Run Script Manually via Railway Shell

```bash
# Open shell in your Railway service
railway run

# Inside the shell, run:
python init_scripts/index_knowledge_base.py
```

### Method 3: Local Re-index (then push to production DB)

⚠️ **Warning:** This requires access to production DATABASE_URL

```bash
# Set up environment variables
export LLM_PROVIDER=nvidia
export NVIDIA_API_KEY=nvapi-...
export EMBEDDING_PROVIDER=nvidia
export EMBEDDING_MODEL=nvidia/nv-embed-v1
export DATABASE_URL=postgresql://...  # from Railway

# Run indexing
cd /path/to/project
python init_scripts/index_knowledge_base.py
```

## Verification

Check Railway logs after re-indexing:

```bash
railway logs
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

Just trigger a re-deployment and the error will be resolved.

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

This saves API quota on every deployment when KB is unchanged.
