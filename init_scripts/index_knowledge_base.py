import asyncio
import logging
import bot.env
import os
import hydra
from hydra.utils import call, instantiate
from omegaconf import DictConfig
import hashlib
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(config: DictConfig) -> None:
    asyncio.run(index(config))

async def index(config: DictConfig) -> None:
    logger.info("Initializing pipeline and loader...")
    pipeline = instantiate(config["pipeline"])
    url_loader = call(config["knowledge"]["loader"])
    doc_transformator = call(config["knowledge"]["transform"])
    
    # --- Global Hash Sync Logic ---
    logger.info("Checking knowledge base state...")
    kb_dir = "knowledge_base"
    # Collect ALL paths first, then sort globally (not just within each dir)
    # os.walk() directory order differs between macOS and Linux!
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

    db_url = config["bot_db_connection"]
    engine = create_engine(db_url)
    
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS kb_sync_state (id INT PRIMARY KEY, hash TEXT)"))
        result = conn.execute(text("SELECT hash FROM kb_sync_state WHERE id = 1")).fetchone()
        old_hash = result[0] if result else None

    if old_hash == current_hash:
        # We also need to confirm that Elasticsearch actually has the index!
        # It's possible for the hash to match but ES was restarted/wiped.
        force_reindex = False
        try:
            retrievers = getattr(pipeline.pipe_retriever, "_child_retrievers", [])
            sparse = retrievers[1] if len(retrievers) > 1 else None
            if sparse and not sparse.client.indices.exists(index=sparse.index_name):
                logger.info(f"Elasticsearch index '{sparse.index_name}' is missing! Forcing re-index.")
                force_reindex = True
        except Exception as e:
            logger.warning(f"Could not check ES index existence: {e}")
            
        if not force_reindex:
            logger.info("✅ Knowledge base is unchanged since last deployment. Skipping indexing to save API limits.")
            return

        
    logger.info("🔄 Knowledge base has changed. Clearing existing indexes and re-indexing...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS docstore CASCADE"))
        conn.commit()
    
    # Extract retrievers to clear/recreate structures
    try:
        # EnsembleRetriever stores children in _child_retrievers
        retrievers = getattr(pipeline.pipe_retriever, "_child_retrievers", [])
        if not retrievers:
            dense = pipeline.pipe_retriever
            sparse = None
        else:
            dense = retrievers[0]
            sparse = retrievers[1] if len(retrievers) > 1 else None
        
        # Recreate docstore schema
        await dense.docstore.acreate_schema()
        
        # Clear Elasticsearch
        sparse.client.indices.delete(index=sparse.index_name, ignore_unavailable=True)
    except Exception as e:
        logger.warning(f"Failed to clear some indexes, proceeding anyway: {e}")

    # --- End Global Hash Sync Logic ---
    
    # Load all markdown documents from knowledge_base
    logger.info("Loading documents from knowledge_base/...")
    from crag.knowledge.loaders.markdown_loader import load
    docs = load("knowledge_base")
    
    if not docs:
        logger.warning("No documents found in knowledge_base/!")
        return
        
    logger.info(f"Loaded {len(docs)} documents. Applying transformations...")
    prepared_docs = doc_transformator.apply(docs)
    
    logger.info(f"Adding {len(prepared_docs)} chunks to the vector store and document store in batches...")
    
    batch_size = 100
    all_ids = []
    for i in range(0, len(prepared_docs), batch_size):
        batch = prepared_docs[i : i + batch_size]
        logger.info(f"Adding batch {i//batch_size + 1} ({len(batch)} chunks)...")
        ids = await pipeline.pipe_retriever.aadd_documents(batch)
        all_ids.extend(ids)
        if i + batch_size < len(prepared_docs):
            logger.info("Sleeping 2s to avoid rate limits...")
            await asyncio.sleep(2)


    
    logger.info(f"Successfully indexed knowledge base. IDs: {len(all_ids)} chunks added.")
    
    # Save the new hash state
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO kb_sync_state (id, hash) VALUES (1, :hash) ON CONFLICT (id) DO UPDATE SET hash = EXCLUDED.hash"), 
            {"hash": current_hash}
        )

if __name__ == "__main__":
    main()
