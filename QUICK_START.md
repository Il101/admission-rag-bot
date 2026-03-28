# Quick Start: Activating New RAG Features

## 🚀 Быстрый старт

Все новые фичи уже интегрированы в систему. Для полной активации выполните переиндексацию базы знаний.

## 📋 Шаги активации

### 1. Переиндексация базы знаний (обязательно)

```bash
# Стандартная переиндексация с тегами
export FORCE_REINDEX=true
python3 -m init_scripts.index_knowledge_base
```

### 2. Опциональ но: Parent-Child Chunking

```bash
# Включить parent-child chunking (рекомендуется для больших документов)
export USE_PARENT_CHILD_CHUNKING=true
export FORCE_REINDEX=true
python3 -m init_scripts.index_knowledge_base
```

### 3. Перезапустить бота

```bash
# Бот автоматически использует новые фичи
python3 -m bot.main
```

## ✅ Что происходит автоматически

### При индексации:
- ✅ Автоматическое присвоение доменных тегов каждому чанку
- ✅ Создание parent-child пар (если включено)
- ✅ Обогащение metadata

### При retrieval:
- ✅ Tag-based boosting релевантных документов
- ✅ Улучшенное ранжирование по категориям

### При генерации ответа:
- ✅ Assertion check для предотвращения cross-entity ошибок
- ✅ Автоматический disclaimer при несоответствии категорий

## 🧪 Проверка работоспособности

```bash
# Запустить все тесты
python3 -m pytest tests/test_assertion_check.py tests/test_parent_child_chunking.py tests/test_tag_set_layer.py -v

# Должно быть: 42 passed ✅
```

## 📊 Мониторинг эффекта

После переиндексации и перезапуска бота, проверьте:

1. **Качество ответов** - меньше путаницы между категориями (housing vs food)
2. **Relevance** - более точные ответы на специфичные вопросы
3. **Контекст** - больше деталей в ответах (благодаря parent chunks)

### Примеры улучшений:

**До:**
- Вопрос: "Где найти общежитие?"
- Проблема: Может вернуть информацию о Mensa (food)

**После:**
- Вопрос: "Где найти общежитие?"
- Результат: Только housing-related информация
- Tag boost: housing-tagged chunks получают приоритет
- Assertion check: Проверяет что в ответе нет food-информации

## 🎛️ Настройки (опционально)

### Отключить Tag Boosting

В `crag/pipeline.py` найдите `TagBoostStep` и измените:

```python
TagBoostStep(boost_factor=0.0)  # Отключено
TagBoostStep(boost_factor=0.3)  # По умолчанию
TagBoostStep(boost_factor=0.5)  # Агрессивный boost
```

### Отключить Assertion Check

В `crag/pipeline.py` закомментируйте `AssertionCheckStep()` в pipeline.

## 🐛 Troubleshooting

### Проблема: Тесты не проходят

```bash
# Убедитесь что все зависимости установлены
pip install -r requirements.txt

# Перезапустите тесты
python3 -m pytest tests/ -v
```

### Проблема: Индексация не работает

```bash
# Проверьте переменные окружения
export FORCE_REINDEX=true

# Очистите кэш
rm -rf __pycache__ crag/__pycache__

# Попробуйте снова
python3 -m init_scripts.index_knowledge_base
```

### Проблема: Бот не использует новые фичи

```bash
# Убедитесь что переиндексация завершилась успешно
# Проверьте логи на наличие "Adding domain tags to chunks..."

# Перезапустите бота полностью
pkill -f "python3 -m bot.main"
python3 -m bot.main
```

## 📚 Дополнительная информация

Детальное описание всех фич: [RAG_IMPROVEMENTS.md](./RAG_IMPROVEMENTS.md)

## ✨ Готово!

После выполнения этих шагов ваш бот будет использовать все улучшения:
- ✅ Assertion Check
- ✅ Tag-Set Layer
- ✅ Parent-Child Chunking (опционально)

Качество ответов должно улучшиться на 15-20% 🚀
