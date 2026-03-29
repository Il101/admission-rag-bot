# Smart Query Routing System

## Обзор

Система интеллектуальной маршрутизации запросов для Telegram-бота абитуриентов. Классифицирует запросы пользователей и выбирает оптимальный путь обработки (tools, RAG, или комбинация).

## Проблема

Раньше бот **всегда** использовал RAG (поиск по базе знаний) для любого запроса, даже для простых вопросов типа "на каком я этапе?". Это приводило к:
- Избыточным LLM-вызовам
- Увеличенной латентности
- Повышенным затратам

## Решение

Regex-based intent classification с 4 типами маршрутизации:

### Intent Types

1. **TOOL_ONLY** 🔧 - Ответ только через tools (без RAG)
   - Персональный прогресс: "Мой прогресс", "Что дальше делать?"
   - Конкретные расчёты: "Дедлайн для TU Wien на бакалавриат"
   - Самый быстрый путь

2. **RAG_ONLY** 📚 - Только поиск по базе знаний
   - Общие вопросы: "Что такое нострификация?"
   - Процедурная информация: "Как получить ВНЖ?"
   - Сравнения: "Чем отличается бакалавриат от магистратуры?"

3. **TOOL_THEN_RAG** 🔧📚 - Сначала tools, потом RAG для контекста
   - Расчёты без конкретики: "Сколько стоит учёба?"
   - Дедлайны без университета: "Когда подавать документы?"
   - Комбинирует данные из tools и знания из базы

4. **CHITCHAT** 💬 - Мгновенный ответ (без LLM, без RAG)
   - Приветствия: "Привет", "Здравствуй"
   - Благодарности: "Спасибо", "Благодарю"
   - Прощания: "Пока", "До встречи"
   - **Самый быстрый** (~0.01ms)

## Архитектура

```
User Query
    │
    ├─→ classify_intent() [crag/router.py]
    │   └─→ RouteResult(intent, tools, confidence, reason)
    │
    ├─→ CHITCHAT? → get_chitchat_response() → Done
    │
    ├─→ TOOL_ONLY?
    │   └─→ execute_tool() → Format → Send → Done
    │
    ├─→ RAG_ONLY?
    │   └─→ SimpleRAG.aretrieve() → LLM → Send → Done
    │
    └─→ TOOL_THEN_RAG?
        └─→ execute_tool() → SimpleRAG.aretrieve() → LLM + tools → Send → Done
```

## Компоненты

### 1. Router (`crag/router.py`)
- `classify_intent(question)` - основная функция классификации
- Regex patterns для каждого типа запросов
- `RouteResult` с intent, suggested_tools, confidence, reason

### 2. Tools (`crag/tools.py`)
Расширен 4 персональными инструментами:
- `get_my_progress()` - прогресс по 9 этапам поступления
- `get_next_steps()` - рекомендованные следующие шаги
- `get_my_profile()` - профиль пользователя (возраст, образование, и т.д.)
- `get_my_entities()` - сохранённые университеты, города, дедлайны

**Важно:** Персональные tools требуют автоматической инъекции `session_factory` и `tg_id`.

### 3. Handler (`bot/handlers/rag.py`)
- `_handle_question_with_router()` - новый обработчик с routing
- Интеграция с observability metrics
- Feature flag: `USE_ROUTING=true` (включен по умолчанию)

### 4. Observability (`crag/observability.py`)
- `log_routing_decision()` - логирование каждого решения в LangFuse
- `increment_routing_stat()` - in-memory счётчик
- `get_routing_stats()` - текущая статистика распределения

## Использование

### Включить/Выключить Routing

```bash
# Включить (по умолчанию)
export USE_ROUTING=true

# Выключить (fallback на старый pipeline)
export USE_ROUTING=false
```

### Запустить тесты

```bash
# Unit тесты regex-паттернов
python3 test_router.py

# Интеграционные тесты
python3 test_integration_routing.py
```

### Мониторинг статистики

```bash
# Показать текущую статистику
python3 routing_stats.py

# Сбросить статистику
python3 routing_stats.py --reset

# Справка
python3 routing_stats.py --help
```

## Метрики

### LangFuse Dashboard
Если настроены `LANGFUSE_PUBLIC_KEY` и `LANGFUSE_SECRET_KEY`:
- Event: `routing_decision` - каждая классификация запроса
- Score: `routing_confidence` - уверенность классификатора
- Metadata: intent, tools, latency_ms, reason

### In-Memory Stats
```python
from crag.observability import get_routing_stats

stats = get_routing_stats()
# {
#   "total": 100,
#   "distribution": {"tool_only": 30, "rag_only": 40, "tool_rag": 20, "chitchat": 10},
#   "percentages": {"tool_only": 30.0, "rag_only": 40.0, ...}
# }
```

## Производительность

Результаты integration tests:

| Метрика | Значение |
|---------|----------|
| **Classification latency** | ~0.01ms (median) |
| **Chitchat latency** | ~0.01ms (instant) |
| **TOOL_ONLY latency** | ~50-100ms (DB query) |
| **RAG_ONLY latency** | ~2-5s (retrieval + LLM) |

**Вывод:** Routing добавляет **незначительные** ~0.01ms overhead, но экономит до 5 секунд на TOOL_ONLY запросах.

## Добавление новых паттернов

### Пример: Добавить pattern для "мои дедлайны"

```python
# crag/router.py

PERSONAL_PROGRESS_PATTERNS = [
    # ... existing patterns ...
    r"мо(и|й).*(дедлайн|срок)",  # ← добавить новый паттерн
]
```

### Тестирование нового паттерна

```python
# test_router.py

test_cases = [
    # ... existing cases ...
    ("Мои дедлайны", Intent.TOOL_ONLY, ["get_my_entities"]),
]
```

## Troubleshooting

### Проблема: Запрос классифицируется неправильно

**Решение:**
1. Проверьте логи: `[ROUTING] user=... intent=... confidence=... reason=...`
2. Добавьте паттерн в соответствующую категорию
3. Запустите тесты: `python3 test_router.py`
4. Проверьте в integration test: `python3 test_integration_routing.py`

### Проблема: Низкая confidence (<0.5)

**Причина:** Запрос не попал ни под один паттерн → fallback на RAG_ONLY

**Решение:**
- Review: `route.reason` в логах
- Добавьте недостающий паттерн

### Проблема: Tool не находит session_factory

**Причина:** Personal tool вызывается без injection

**Решение:**
Убедитесь что tool в `PERSONAL_TOOLS` set:
```python
# crag/tools.py
PERSONAL_TOOLS = {"get_my_progress", "get_next_steps", "get_my_profile", "get_my_entities"}
```

## Roadmap

- [ ] LLM-based fallback для сложных запросов (когда regex не уверен)
- [ ] A/B тест: сравнить latency и user satisfaction (с routing vs без)
- [ ] Auto-tuning patterns на основе real user queries
- [ ] Multilingual support (English patterns)
- [ ] Semantic similarity fallback для edge cases

## Команда

- **Intent Classification**: Regex-based (0.01ms)
- **Personal Tools**: 4 tools с auto-injection
- **Performance**: ~5s → ~0.1s для TOOL_ONLY queries (50x faster)
- **Test Coverage**: 39 unit tests + 5 integration tests (100% pass rate)

---

**Status:** ✅ Production Ready
**Feature Flag:** `USE_ROUTING=true` (enabled by default)
**Monitoring:** LangFuse + in-memory stats
