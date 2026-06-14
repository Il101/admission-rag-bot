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
- A single, feature-rich RAG pipeline (`SimpleRAG`) combining dense vector search with PostgreSQL full-text search
- Query rewriting, LLM-based document grading, optional reranking, and HyDE (Hypothetical Document Embeddings)
- Semantic answer caching and an intent/query router (fact vs. narrative vs. tool-based questions)
- Pluggable LLM/embedding providers (Google Gemini, OpenAI, NVIDIA / OpenAI-compatible APIs) via environment variables

## ✨ Features

### 🤖 RAG Pipeline (SimpleRAG)
- **Hybrid retrieval**: dense vector similarity search (pgvector) combined with PostgreSQL native full-text search ("BM25-like" ranking)
- **Query rewriting**: automatic reformulation of queries that don't retrieve relevant results
- **LLM-based document grading**: batch relevance scoring of retrieved chunks before they're used in the answer
- **HyDE**: generates a hypothetical answer to improve retrieval for hard queries
- **Optional reranking** of retrieved documents
- **Semantic answer cache**: caches full answers keyed by embedding similarity of the question, skipping the full pipeline on near-duplicate questions
- **Query router**: classifies questions as fact (structured YAML data), narrative (markdown guides), tool-based, or chit-chat, and routes them accordingly
- **Tool calling**: lets the LLM call tools (e.g., for deterministic lookups) in addition to retrieval

### 🔍 Hybrid Search
- **Dense vector search**: semantic search over embeddings stored in pgvector
- **Full-text search**: PostgreSQL `tsvector`/`tsquery`-based keyword search
- Results from both are combined with weighted scoring (`VECTOR_WEIGHT` / `FTS_WEIGHT` in `crag/simple_rag.py`)

### 🧠 LLM & Embedding Providers
- **Google Gemini** (default): `gemini-2.5-flash`, with native embeddings (`gemini-embedding-001`)
- **OpenAI-compatible APIs**: OpenAI, NVIDIA, OpenRouter and similar providers, selected via `LLM_PROVIDER`
- Switching providers is done via environment variables (see `docs/LLM_PROVIDERS.md`), not via Hydra config groups

### 🇺🇦 Multilingual Support
- Knowledge base facts and prompts support multiple languages (Russian/Ukrainian/German/English)
- LLM-based answer generation handles the user's language directly — no separate local translation/embedding models are used

### 🛠 Production-Ready
- Docker Compose deployment (bot + PostgreSQL/pgvector + pgAdmin)
- PostgreSQL with the pgvector extension for vector storage
- Structured fact-based knowledge base (`facts/`) plus markdown narrative guides
- Admin commands for knowledge base management
- Optional Langfuse observability integration

## 🏗 Architecture

### RAG Pipeline

The live implementation is the `SimpleRAG` class (`crag/simple_rag.py`), which combines several techniques into a single pipeline:

```
Query → Query Router (fact / narrative / tool / chitchat)
      → [optional] Query Rewriting / HyDE
      → Hybrid Retrieval (pgvector similarity + PostgreSQL full-text search)
      → LLM Document Grading (batch relevance scoring)
      → [optional] Reranking
      → Semantic Cache check
      → LLM → Answer
```

If no relevant documents are found, the pipeline can rewrite the query and retry retrieval (bounded number of attempts) before falling back to a "no answer found" message.

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
cp example.env .env
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
│   ├── handlers/          # Command handlers (rag, management, onboarding, ...)
│   ├── middlewares/        # Bot middlewares
│   └── filters/            # Message filters
├── crag/                  # RAG pipeline implementation
│   ├── simple_rag.py      # SimpleRAG: hybrid retrieval, grading, HyDE, caching
│   ├── llm_providers.py   # LLM/embedding provider abstraction (Google, OpenAI, NVIDIA)
│   ├── query_router.py    # Fact vs. narrative query classification
│   ├── router.py          # Intent routing (tool / RAG / chitchat)
│   ├── pipeline.py        # Pipeline steps and guardrails
│   ├── yaml_facts_indexer.py  # Indexes facts/*.yaml into pgvector
│   ├── parent_child_chunking.py  # Optional parent-child chunk generation
│   ├── tag_set_layer.py   # Domain tag boosting for retrieval
│   ├── assertion_validator.py  # Post-generation cross-entity checks
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
