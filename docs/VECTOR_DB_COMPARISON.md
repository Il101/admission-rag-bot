# Vector Database Comparison for Admission Bot

## Current: PostgreSQL + pgvector

**Pros:**
- Unified storage (users, chat, vectors in one DB)
- Hybrid search (vector + FTS) out of the box
- JSONB metadata filtering with GIN indexes
- Simple Railway deployment (one service)
- Cost: $0 extra (included in PostgreSQL)
- Retrieval: ~0.4s for 600 chunks

**Cons:**
- HNSW index limited to 2000 dims (current: 1536-3072)
- Sequential scan for large vectors (slower than specialized DBs)
- No advanced features (quantization, multi-vector)

---

## Alternative 1: Qdrant

**Pros:**
- Open source, self-hosted or cloud
- Excellent metadata filtering with payload storage
- Hybrid sparse/dense vectors
- Quantization for memory efficiency
- Optimized for retrieval (10-50ms for similar dataset)
- Scroll API for pagination
- Multi-tenancy support

**Cons:**
- Additional service to manage (Railway or separate host)
- Cost: Free (self-hosted) or $25-50/mo (cloud)
- Need to sync metadata (users, chat) separately with PostgreSQL
- Migration effort: rewrite indexing + retrieval code

**When to use:**
- >10K chunks
- Need <100ms retrieval latency
- Multi-tenant scenario (thousands of users with personal KBs)
- Advanced filtering requirements

---

## Alternative 2: Pinecone

**Pros:**
- Managed cloud (zero ops)
- Very fast (5-20ms)
- Good metadata filtering
- Serverless tier (pay-per-use)

**Cons:**
- Cost: $70-100/mo for production (1M vectors)
- Vendor lock-in
- Still need PostgreSQL for users/chat
- No hybrid search (vector only)

**When to use:**
- Budget is not a constraint
- Want zero infrastructure management
- Need global CDN-like latency

---

## Alternative 3: Weaviate

**Pros:**
- Open source + cloud options
- GraphQL API (nice for flexible queries)
- Built-in vectorization modules
- Good documentation

**Cons:**
- More complex setup than Qdrant
- Heavier resource usage
- Overkill for small KB

---

## Alternative 4: ChromaDB

**Pros:**
- Lightweight, embeddable
- Simple API
- Good for small projects
- Can run in-process (no separate service)

**Cons:**
- Less production-ready than Qdrant
- Limited scalability
- Fewer advanced features

---

## Recommendation for Current Project

### STAY with PostgreSQL + pgvector

**Why:**
1. **Size**: 600 chunks is tiny. PostgreSQL handles this easily.
2. **Performance**: 0.4s retrieval is acceptable for a Telegram bot.
3. **Simplicity**: One service, one connection string, unified backups.
4. **Cost**: $0 extra infrastructure.

**Current problems are NOT database performance:**
- ❌ Facts not found → chunking strategy issue
- ❌ Ellipsis responses → validation issue
- ❌ VWU confusion → document structure issue

### When to Consider Migration to Qdrant

Migrate IF any of these apply:
- [ ] >10K chunks (100x growth)
- [ ] Retrieval latency >1s (2.5x slower)
- [ ] Need multi-tenant with isolated collections per user
- [ ] Need advanced features (multi-modal, graph search)
- [ ] Planning to scale to hundreds of documents

### Migration Path (if needed)

If you do decide to migrate later:

```python
# 1. Index to Qdrant (parallel to pgvector)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="admission_kb",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# 2. Migrate data
for chunk in chunks:
    client.upsert(
        collection_name="admission_kb",
        points=[
            PointStruct(
                id=chunk["id"],
                vector=chunk["embedding"],
                payload={
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                }
            )
        ]
    )

# 3. Update SimpleRAG.aretrieve() to use Qdrant
async def aretrieve(self, query: str, top_k: int = 6, user_filters: dict = None):
    embedding = await self.get_embedding(query)

    filter_conditions = []
    if user_filters:
        if "country_scope" in user_filters:
            filter_conditions.append(
                {"key": "metadata.country_scope", "match": {"value": user_filters["country_scope"]}}
            )

    results = client.search(
        collection_name="admission_kb",
        query_vector=embedding,
        limit=top_k,
        query_filter={"must": filter_conditions} if filter_conditions else None,
    )

    docs = [
        Document(page_content=hit.payload["content"], metadata=hit.payload["metadata"])
        for hit in results
    ]
    return docs
```

**Effort:** ~1-2 days
**Cost:** +$0 (self-hosted on Railway) or +$25/mo (Qdrant Cloud)

---

## Conclusion

**For current scale (600 chunks):**
- PostgreSQL + pgvector is optimal
- Focus on fixing chunking & document structure
- Save Qdrant migration for when you hit 10K+ chunks

**For future scale (10K+ chunks, multi-tenant):**
- Qdrant is the best next step
- Pinecone if budget allows and you want zero ops
- Weaviate if you need graph features
