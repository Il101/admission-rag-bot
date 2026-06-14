# admission-rag-bot

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Telegram Bot](https://img.shields.io/badge/Telegram-Bot%20API-blue?logo=telegram)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-316192?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> 🎓 AI-powered Telegram assistant helping students with university admission questions using Retrieval-Augmented Generation

An intelligent Telegram bot built with **Retrieval-Augmented Generation (RAG)** to automatically answer frequently asked questions about university admission, visa processes, housing, scholarships, and student life in Austria. Reduces volunteer workload by providing instant, accurate responses from a curated knowledge base.

## Why This Bot?

Built to reduce volunteer workload during university admission season. Handles FAQ about applications, visa processes, housing, and student life — so volunteers can focus on complex individual cases.

**Technical highlights:**
- An intent-router-driven RAG pipeline with **strict grounding** — it answers only from retrieved context and gives an honest "no answer" when nothing relevant is found
- **Source citations**: every factual statement is tagged with a `[n]` marker and the answer ends with a deduplicated source list built from the actually-retrieved facts (no model-invented URLs)
- **Faithfulness gate**: an LLM-judge verifies the answer against its sources and triggers a single bounded regeneration (or a caution) when a claim isn't supported
- Hybrid retrieval (dense pgvector + PostgreSQL full-text), LLM-based document grading, optional **cross-encoder reranking** (Cohere / Jina / NVIDIA), and HyDE for narrative queries
- **Effective-dated facts**: time-sensitive facts carry a `valid_for` academic year and a `source_url`, surfaced as "актуально для приёма 2026/27" with escalation to the official source for high-stakes topics
- Pluggable LLM/embedding providers (Google Gemini, OpenAI, NVIDIA / OpenAI-compatible APIs) via environment variables

## ✨ Features

### 🤖 RAG Pipeline
- **Strict grounding**: answers only from retrieved context; returns an honest "no relevant information found" instead of fabricating when grading filters out all documents
- **Hybrid retrieval**: dense vector similarity search (pgvector) combined with PostgreSQL native full-text search ("BM25-like" ranking)
- **Query rewriting**: automatic reformulation of the query before retrieval (anaphora resolution, keyword enrichment)
- **LLM-based document grading**: batch relevance scoring of retrieved chunks before they're used in the answer
- **HyDE**: generates a hypothetical document to improve retrieval for narrative queries (disabled for fact lookups to avoid biasing factual retrieval)
- **Optional reranking** of retrieved documents, plus domain-tag boosting
- **Semantic answer cache**: caches answers keyed by question-embedding similarity and scoped by user profile; skipped for fact-type questions so a near-duplicate question about a different university/category never reuses a stale answer
- **Query router**: classifies questions as fact (structured YAML data), narrative (markdown guides), tool-based, or chit-chat, and routes them accordingly
- **KB-backed tools**: budget and date calculations and personal-progress lookups; tool data is read from the `facts/` knowledge base, not hardcoded

### 🛡 Reliability & Trust

Designed for a high-stakes domain (deadlines, fees, visas) where a wrong answer is costly. Five layers reduce hallucination:

1. **Strict grounding** — generation runs only over retrieved context; if document grading filters everything out, the bot returns an honest "no relevant information found" instead of guessing. Internal reasoning steps (grading, reranking, rewriting) run at temperature 0.
2. **Source citations** — context facts are numbered, the model must cite the `[n]` it used, and the bot appends a deduplicated "📎 Источники" list built from the real retrieved documents (`crag/simple_rag.py::build_numbered_context` + `bot/utils.py::docs_to_sources_str` share one numbering).
3. **Faithfulness verification** — after generation, `verify_faithfulness` (LLM-judge, temperature 0) checks every claim against the sources; unsupported claims trigger one stricter regeneration, then a visible caution if still unverified (`crag/assertion_validator.py`, `AssertionCheckStep`).
4. **Cross-encoder reranking** — optional dedicated rerank API (Cohere / Jina / NVIDIA) for sharper top-k ordering, with graceful fallback to the built-in LLM reranker when unconfigured (`crag/reranker.py`).
5. **Effective-dated, KB-backed facts** — time-sensitive facts carry `valid_for` + `source_url`; tools read from `facts/` rather than hardcoded values, and high-stakes answers escalate to the official source when data is missing or may be outdated.

### 🔍 Hybrid Search
- **Dense vector search**: semantic search over embeddings stored in pgvector
- **Full-text search**: PostgreSQL `tsvector`/`tsquery`-based keyword search
- Results from both are combined with weighted scoring (`VECTOR_WEIGHT` / `FTS_WEIGHT` in `crag/simple_rag.py`)

### 🧠 LLM & Embedding Providers
- **Google Gemini** (default): `gemini-2.5-flash`, with native embeddings (`gemini-embedding-001`)
- **OpenAI-compatible APIs**: OpenAI, NVIDIA, OpenRouter and similar providers, selected via `LLM_PROVIDER`
- Switching providers is done via environment variables (see `docs/LLM_PROVIDERS.md`), not via Hydra config groups

### 🌐 Multilingual Support
- Knowledge base facts and prompts mix Russian, German and English source material
- LLM-based answer generation handles the user's language directly — no separate local translation/embedding models are used

### 🛠 Production-Ready
- Docker Compose deployment (bot + PostgreSQL/pgvector + pgAdmin)
- PostgreSQL with the pgvector extension for vector storage
- Structured fact-based knowledge base (`facts/`) plus markdown narrative guides
- Admin commands for knowledge base management
- Optional Langfuse observability integration

## 🏗 Architecture

### RAG Pipeline

The live request flow is an intent **router** (`crag/router.py`) feeding a step-based **pipeline** (`crag/pipeline.py`) that orchestrates the retrieval/grading/generation operations implemented in `SimpleRAG` (`crag/simple_rag.py`):

```
Query → Intent Router (RAG / tool+RAG / personal / chitchat)
      → Semantic answer-cache check (skipped for fact lookups)
      → Query Rewriting  (+ HyDE for narrative queries only)
      → Hybrid Retrieval (pgvector similarity + PostgreSQL full-text search)
      → LLM Document Grading (batch relevance scoring)
      → [optional] Cross-encoder reranking + domain-tag boosting
      → Numbered context build → answer generation (with [n] citations)
      → Faithfulness check → regenerate once if unsupported → append sources
```

**Strict grounding:** if grading finds no relevant documents, the pipeline stops and returns an honest "no relevant information found" message instead of answering from the model's own knowledge. Internal reasoning steps (grading, reranking, rewriting) run at temperature 0 for stability.

### Technology Stack

**Backend Framework**
- Python 3.11+
- `python-telegram-bot` (Telegram Bot API)
- SQLAlchemy (async) for database access

**Vector Search & Retrieval**
- PostgreSQL 16 with the `pgvector` extension (`pgvector/pgvector:pg16` image)
- Hybrid search: pgvector similarity search + PostgreSQL native full-text search (no separate search engine)

**LLM & Embeddings**
- Google Gemini API (`gemini-2.5-flash`, default) — also used for embeddings (`gemini-embedding-001`)
- OpenAI-compatible APIs (OpenAI, NVIDIA, OpenRouter, etc.) as alternative LLM/embedding providers
- All LLM and embedding calls go through external APIs — there is no local model inference (no llama.cpp, no SentenceTransformers, no torch)

**Configuration & Orchestration**
- Hydra/OmegaConf for prompt configuration (`configs/default.yaml` + `configs/prompts/`)
- Docker Compose (deployment)
- Optional Langfuse for LLM observability/tracing

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))
- An API key for at least one LLM provider (Google Gemini, OpenAI, or NVIDIA)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Il101/admission-rag-bot.git
cd admission-rag-bot
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings:
# - TGBOT_TOKEN: Your Telegram bot token
# - GOOGLE_API_KEY (default provider) or OPENAI_API_KEY / NVIDIA_API_KEY
# - POSTGRES_* variables for the database connection
```

3. **Start services**
```bash
docker compose up -d
```
This starts the bot, a `pgvector/pgvector:pg16` PostgreSQL database, and pgAdmin.

4. **Index the knowledge base**
The knowledge base lives in `facts/` (structured YAML facts) and is indexed into pgvector via:
```bash
python3 -m init_scripts.index_knowledge_base
```
This step runs automatically as part of the container startup sequence (`init_scripts/entry.sh`), but can be re-run manually after editing facts (see `docs/REINDEX_KB.md`).

The bot will be online and ready to answer questions!

## 💬 Bot Commands

### User Commands
- `/start` - Welcome message and onboarding
- `/help` - Display available commands
- `/ans <question>` - Get AI-generated answer to your question
- `/ans_rep` - Answer question from replied message
- `/docs <question>` - Get relevant documents without LLM answer
- `/docs_rep` - Get documents from replied message
- `/delete_my_data` - Delete your stored data

### Admin Commands
- `/ban <user_id>` / `/unban <user_id>` - Ban/unban a user
- `/add_admin` - Grant admin rights
- `/add` / `/del` - Add or remove a knowledge base fact
- `/stats` - Show usage statistics

*Admin commands require an authorized Telegram ID configured via `FATHER_TG_ID`.*

## ⚙️ Configuration

The bot uses [Hydra](https://hydra.cc/)/OmegaConf for a small amount of configuration, primarily prompt selection. The actual config tree is minimal:

```
configs/
├── default.yaml          # Top-level config: selects the active prompt set and DB connection string
└── prompts/               # Prompt templates per model
    ├── gemini-2.5-flash.yaml
    ├── chatgpt-4o-mini.yaml
    └── gemma2.yaml
```

`configs/default.yaml` selects which prompt file is active (default: `gemini-2.5-flash`) and builds the PostgreSQL connection string from environment variables. There are **no** `configs/llm/`, `configs/retriever/`, `configs/pipeline/`, or `configs/knowledge/` config groups — LLM provider selection, retrieval behavior, and pipeline behavior are controlled via environment variables and code in `crag/`, not via Hydra config groups.

### Switching LLM / Embedding Providers

Provider selection is done via environment variables (see `docs/LLM_PROVIDERS.md` for details), e.g.:

```bash
# .env
LLM_PROVIDER=google          # google (default) | openai | nvidia
GOOGLE_API_KEY=your_key_here
```

### Switching Prompts

To use a different prompt set, change the `prompts` default in `configs/default.yaml`:
```yaml
defaults:
  - _self_
  - prompts: gemini-2.5-flash   # or: chatgpt-4o-mini, gemma2
```

## 📊 Project Structure

```
admission-rag-bot/
├── bot/                   # Telegram bot implementation
│   ├── handlers/          # Command/callback handlers (rag, management, onboarding, ...)
│   ├── db.py              # SQLAlchemy (async) models and queries
│   ├── memory.py          # Per-user journey state and conversation memory
│   └── app.py             # Bot entrypoint and handler registration
├── crag/                  # RAG pipeline implementation
│   ├── simple_rag.py      # SimpleRAG: hybrid retrieval, grading, HyDE, caching
│   ├── llm_providers.py   # LLM/embedding provider abstraction (Google, OpenAI, NVIDIA)
│   ├── query_router.py    # Fact vs. narrative query classification
│   ├── router.py          # Intent routing (tool / RAG / chitchat)
│   ├── reranker.py        # Optional cross-encoder rerank API (Cohere/Jina/NVIDIA)
│   ├── pipeline.py        # Pipeline steps + grounding/citation/faithfulness guardrails
│   ├── yaml_facts_indexer.py  # Indexes facts/*.yaml into pgvector
│   ├── parent_child_chunking.py  # Optional parent-child chunk generation
│   ├── tag_set_layer.py   # Domain tag boosting for retrieval
│   ├── assertion_validator.py  # Faithfulness verification (LLM-judge)
│   ├── ab_testing.py      # A/B testing helpers
│   ├── observability.py   # Langfuse tracing helpers
│   └── tools.py           # Tool-calling implementations
├── facts/                 # Structured knowledge base (YAML)
│   ├── universities/       # University-specific facts (deadlines, fees, etc.)
│   ├── language/           # Language requirements, course providers
│   └── financial/          # Cost-of-living / budget facts
├── configs/               # Hydra configuration (default.yaml + prompts/)
├── init_scripts/          # Startup, DB init, and indexing scripts
├── scripts/               # Utility scripts (ingestion, KB freshness checks)
├── schema/                # Documentation of the facts/ YAML schema
├── tests/                 # Unit tests
├── docker-compose.yml     # Docker orchestration (bot + pgvector + pgAdmin)
└── docker-compose.release.yml  # Compose file using a prebuilt image
```

## 🔧 Development

### Running Tests
```bash
pytest tests/
```

### Adding Knowledge

The active knowledge base is the set of structured YAML facts in `facts/` (see `schema/README.md` for the schema):
```
facts/
├── universities/   # University-specific facts (deadlines, tuition, language requirements, ...)
├── language/        # Language course providers and certification rules
└── financial/       # Cost-of-living and budget facts
```

After adding or editing facts, re-index the knowledge base:
```bash
python3 -m init_scripts.index_knowledge_base
```

`scripts/ingest_all.py` is a scraping/ingestion helper for generating narrative markdown source documents from external sites, and `scripts/verify_kb_freshness.py` checks whether those source documents have changed since the last ingestion. The indexer also scans a `knowledge_base/` directory for markdown content if present; in this repo the active, indexed knowledge base is the structured YAML in `facts/`.

## 📈 Use Cases

- 🎓 **University Admission Support**: Answer questions about applications, requirements, deadlines
- 🏠 **Student Life Guidance**: Housing, budgets, city information
- 🛂 **Visa & Immigration**: D-visa, residence permits, documentation
- 💰 **Financial Planning**: Tuition fees, scholarships, cost of living
- 🌍 **Language Requirements**: German/English requirements, preparatory courses

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Expanding the structured facts knowledge base
- Improving query routing and retrieval ranking
- UI/UX improvements for admin tools

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) - Telegram Bot framework
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search for PostgreSQL

---

**Author**: Ilia Zharikov  
**Contact**: [LinkedIn](https://www.linkedin.com/in/ilia-zharikov)
