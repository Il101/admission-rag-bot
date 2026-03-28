# Настройка LLM Провайдеров

Бот поддерживает несколько LLM провайдеров через унифицированный интерфейс. Выбор провайдера настраивается через переменные окружения.

## 📋 Поддерживаемые провайдеры

### 1. Google Gemini (по умолчанию)

**Настройка:**
```bash
LLM_PROVIDER=google
GOOGLE_API_KEY=your_api_key
GOOGLE_MODEL=gemini-2.0-flash-exp  # или gemini-2.5-flash
```

**Получить API ключ:** https://aistudio.google.com/app/apikey

**Особенности:**
- ✅ Бесплатный уровень с generous limits
- ✅ Нативная поддержка structured outputs (JSON schema)
- ✅ Эмбеддинги включены (gemini-embedding-001)
- ✅ Очень быстрый ответ

**Рекомендуется для:** production использования (бесплатно + надежно)

---

### 2. NVIDIA AI (OpenAI-compatible)

**Настройка:**
```bash
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-Kd4P...
OPENAI_MODEL=openai/gpt-oss-120b
```

**Получить API ключ:** https://build.nvidia.com/

**Доступные модели:**
- `openai/gpt-oss-120b` - GPT clone с reasoning
- `nvidia/llama-3.1-nemotron-70b-instruct` - Llama 3.1 70B
- `meta/llama-3.1-405b-instruct` - Llama 3.1 405B (мощная)
- `mistralai/mixtral-8x22b-instruct-v0.1` - Mixtral 8x22B

**Особенности:**
- ✅ Бесплатный доступ к топовым моделям
- ✅ Поддержка reasoning (для GPT-OSS)
- ⚠️ Эмбеддинги через Google Gemini (требуется GOOGLE_API_KEY)
- ⚠️ Structured outputs через промпт-инжиниринг

**Рекомендуется для:** экспериментов с различными моделями

---

### 3. OpenAI

**Настройка:**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1  # опционально
```

**Получить API ключ:** https://platform.openai.com/api-keys

**Доступные модели:**
- `gpt-4o` - новейшая GPT-4 Omni
- `gpt-4o-mini` - быстрая и дешевая
- `gpt-3.5-turbo` - бюджетная

**Особенности:**
- ✅ Нативные эмбеддинги (text-embedding-3-small)
- ✅ Высокое quality ответов
- 💰 Платная модель (pay-per-token)

**Рекомендуется для:** production с бюджетом

---

## 🔀 Как переключить провайдера

### На Railway:

```bash
# Переключиться на NVIDIA
railway variables set LLM_PROVIDER=nvidia
railway variables set NVIDIA_API_KEY=nvapi-your-key
railway variables set OPENAI_MODEL=openai/gpt-oss-120b

# Для эмбеддингов (обязательно при использовании NVIDIA)
railway variables set GOOGLE_API_KEY=your-google-key
```

### Локально:

Скопируй `example.env` в `.env` и измени:
```bash
cp example.env .env
nano .env  # или vim, code, etc.
```

---

## 🧪 Архитектура абстракции

Все провайдеры реализуют единый интерфейс `BaseLLMProvider`:

```python
class BaseLLMProvider:
    async def generate(prompt: str, system_prompt: str) -> str
    async def generate_stream(prompt: str, system_prompt: str) -> AsyncIterator[str]
    async def embed(texts: List[str]) -> List[List[float]]
```

**Код абстракции:** `crag/llm_providers.py`

**Автоматический выбор:** `LLMProviderFactory.get_provider()` читает `LLM_PROVIDER` из env

---

## 📊 Сравнение провайдеров

| Провайдер | Цена | Скорость | Quality | Embeddings | Structured Output |
|-----------|------|----------|---------|------------|-------------------|
| **Google Gemini** | Бесплатно | ⚡⚡⚡ | ⭐⭐⭐⭐ | Нативные | JSON schema |
| **NVIDIA AI** | Бесплатно | ⚡⚡ | ⭐⭐⭐⭐⭐ | Через Google | Промпт |
| **OpenAI** | Платно | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Нативные | Промпт |

---

## ❓ FAQ

**Q: Можно ли использовать NVIDIA без Google API ключа?**
A: Нет, для эмбеддингов требуется `GOOGLE_API_KEY`. В будущем можно добавить OpenAI embeddings.

**Q: Что будет если не указать LLM_PROVIDER?**
A: Используется Google Gemini по умолчанию.

**Q: Можно ли добавить свой провайдер (например, Anthropic Claude)?**
A: Да! Создай класс-наследник `BaseLLMProvider` в `crag/llm_providers.py` и добавь его в фабрику.

**Q: Как проверить что провайдер работает?**
A: Смотри логи при запуске бота:
```
INFO:crag.llm_providers:Initialized LLM provider: nvidia with model openai/gpt-oss-120b
```

**Q: Влияет ли провайдер на качество RAG?**
A: HyDE, re-ranking, и tool calling работают на любом провайдере. Structured outputs (JSON) у Gemini работают лучше, но промпт-инжиниринг тоже надежен.
