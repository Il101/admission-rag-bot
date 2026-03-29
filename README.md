# FreshmanRAG Bot

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)
![Telegram Bot](https://img.shields.io/badge/Telegram-Bot%20API-blue?logo=telegram)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-316192?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> 🎓 AI-powered Telegram assistant helping students with university admission questions using advanced RAG techniques

An intelligent Telegram bot built with **Retrieval-Augmented Generation (RAG)** to automatically answer frequently asked questions about university admission, visa processes, housing, scholarships, and student life in Austria. Reduces volunteer workload by providing instant, accurate responses from a curated knowledge base.

## ✨ Features

### 🤖 Advanced RAG Pipelines
- **Simple RAG**: Direct document retrieval and answer generation
- **Conditional RAG with Filtering**: LLM-powered document relevance scoring
- **Conditional RAG with Question Rewriting**: Automatic query reformulation for better results

### 🔍 Hybrid Search System
- **Dense Vector Search**: Semantic search using sentence embeddings (pgvector)
- **Sparse BM25 Search**: Keyword-based retrieval (Elasticsearch)
- **Ensemble Retriever**: Combines both approaches with Reciprocal Rank Fusion
- **Parent Document Strategy**: Searches small chunks, returns full context

### 🧠 Flexible LLM Support
- **Gemma2-2B** (default): CPU-optimized quantized model
- **OpenAI GPT models**: Optional cloud integration
- **Google Gemini**: Alternative cloud LLM
- Configurable via Hydra framework

### 🇺🇦 Ukrainian Language Optimization
- Uses multilingual sentence transformers optimized for Ukrainian
- Ukrainian-aware text processing and embeddings

### 🛠 Production-Ready
- Docker Compose deployment
- PostgreSQL with pgvector extension
- Elasticsearch for BM25 search
- Comprehensive configuration system (Hydra)
- Admin commands for knowledge base management

## 🏗 Architecture

### RAG Pipelines

The bot implements three sophisticated RAG pipelines, each with increasing intelligence:

#### 1. Simple RAG
Direct retrieval and generation pipeline:
```
Query → Retriever → Documents → LLM → Answer
```

#### 2. Conditional RAG with Filtering
Adds document relevance scoring:
```
Query → Retriever → Documents → Grading (LLM) → Relevant Docs → LLM → Answer
                                              ↓ (if none relevant)
                                           Give Up Message
```

#### 3. Conditional RAG with Question Rewriting (Default)
Automatically reformulates queries for better results:
```
Query → Retriever → Documents → Grading → Relevant Docs → LLM → Answer
                                       ↓ (if none relevant)
                                  Rewrite Query → Retry (max 3 times)
```

### Technology Stack

**Backend Framework**
- Python 3.11+
- aiogram 3.x (Telegram Bot API)
- LangChain (RAG orchestration)

**Vector Search & Retrieval**
- PostgreSQL 16 with pgvector extension
- Elasticsearch 8.x (BM25 search)
- SentenceTransformers (`lang-uk/ukr-paraphrase-multilingual-mpnet-base`)

**LLM Integration**
- llama.cpp (local inference)
- OpenAI API (optional)
- Google Gemini API (optional)

**Configuration & Orchestration**
- Hydra (configuration management)
- Docker Compose (deployment)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM (for local LLM inference)
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Il101/FreshmanRAG_bot.git
cd FreshmanRAG_bot
```

2. **Download required models**
```bash
bash init_scripts/download_embeddings.sh
bash init_scripts/download_llm.sh  # Optional: for local inference
```

3. **Configure environment**
```bash
cp example.env .env
# Edit .env with your settings:
# - TGBOT_TOKEN: Your Telegram bot token
# - GOOGLE_API_KEY or OPENAI_API_KEY: If using cloud LLMs
```

4. **Prepare data directories**
```bash
bash init_scripts/prepare_data_volumes.sh
```

5. **Start services**
```bash
docker compose up -d
```

The bot will be online and ready to answer questions!

## 💬 Bot Commands

### User Commands
- `/start` - Welcome message and bot introduction
- `/help` - Display available commands
- `/ans <question>` - Get AI-generated answer to your question
- `/ans_rep` - Answer question from replied message
- `/docs <question>` - Get relevant documents without LLM answer
- `/docs_rep` - Get documents from replied message

### Admin Commands
- `/ban <user_id>` - Ban user from using the bot
- `/unban <user_id>` - Unban user
- `/add_fact` - Add information to knowledge base
- `/add_link` - Add public resource link

*Admin commands require authorized Telegram ID in configuration*

## ⚙️ Configuration

The bot uses [Hydra](https://hydra.cc/) for flexible configuration management. All configs are in the `configs/` directory.

### Configuration Structure
```
configs/
├── default.yaml          # Main configuration
├── llm/                  # LLM provider configs
├── retriever/            # Retriever strategy configs
├── prompts/              # System prompts
├── pipeline/             # RAG pipeline configs
└── knowledge/            # Data loading configs
```

### Switching RAG Pipelines

Edit `configs/default.yaml`:
```yaml
defaults:
  - pipeline: simple_rag              # Simple RAG
  # - pipeline: rag_with_filtering    # With filtering
  # - pipeline: rag_with_rewriting    # With rewriting (default)
```

### Using Different LLMs

**OpenAI:**
```yaml
# configs/default.yaml
defaults:
  - llm: openai
```
```bash
# .env
OPENAI_API_KEY=your_key_here
```

**Google Gemini:**
```yaml
defaults:
  - llm: gemini
```
```bash
# .env
GOOGLE_API_KEY=your_key_here
```

## 📊 Project Structure

```
FreshmanRAG_bot/
├── bot/                  # Telegram bot implementation
│   ├── handlers/         # Command handlers
│   ├── middlewares/      # Bot middlewares
│   └── filters/          # Message filters
├── crag/                 # RAG pipeline implementation
│   ├── chains/           # LangChain workflows
│   ├── llm_providers.py  # LLM integrations
│   └── retrievers.py     # Search implementations
├── configs/              # Hydra configuration
├── init_scripts/         # Deployment utilities
├── knowledge_base/       # Knowledge base (not in git)
├── tests/                # Unit tests
└── docker-compose.yml    # Docker orchestration
```

## 🔧 Development

### Running Tests
```bash
pytest tests/
```

### Adding Knowledge
The bot's knowledge base is organized in markdown files (excluded from git for privacy):
```
knowledge_base/
├── universities/         # University-specific info
├── processes/            # Visa, housing, etc.
├── financial/            # Costs, scholarships
└── language/             # Language requirements
```

Add new documents and run ingestion:
```bash
python ingest_all.py
```

## 📈 Use Cases

- 🎓 **University Admission Support**: Answer questions about applications, requirements, deadlines
- 🏠 **Student Life Guidance**: Housing, budgets, city information
- 🛂 **Visa & Immigration**: D-visa, residence permits, documentation
- 💰 **Financial Planning**: Tuition fees, scholarships, cost of living
- 🌍 **Language Requirements**: German/English requirements, preparatory courses

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Fine-tuning LLMs for Ukrainian
- Adding encoder-based document filtering
- Expanding retrieval strategies
- UI/UX improvements for admin tools

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - RAG framework
- [aiogram](https://github.com/aiogram/aiogram) - Telegram Bot framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Local LLM inference
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search

---

**Author**: Ilia Zharikov  
**Contact**: [LinkedIn](https://www.linkedin.com/in/ilia-zharikov)
