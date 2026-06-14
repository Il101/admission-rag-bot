# Инструкция по переиндексации после изменений в базе знаний

## Когда нужна переиндексация

Переиндексация требуется после любых изменений в:
- `facts/` — структурированные YAML-факты (университеты, языковые требования, финансы)
- markdown-документах в `knowledge_base/` (нарративные гайды), если такая директория присутствует в проекте

а также при смене embedding-провайдера или модели (см. `docs/REINDEX_KB.md`).

## Как переиндексировать (Docker Compose)

### Вариант 1: Принудительная переиндексация при запуске контейнера

```bash
# В .env (или переменных окружения перед docker compose up)
export FORCE_REINDEX=true

docker compose up -d --build
```

`init_scripts/entry.sh` запускает индексацию автоматически при каждом старте контейнера бота:
```bash
python3 init_scripts/init_bot_db.py
python3 -m bot.migrate
python3 init_scripts/index_knowledge_base.py
python3 bot/app.py
```

При `FORCE_REINDEX=true` индексация выполняется полностью, независимо от того, изменился ли хэш файлов в `facts/`.

После успешной переиндексации не забудьте убрать `FORCE_REINDEX` из `.env`, чтобы не пересчитывать эмбеддинги при каждом перезапуске.

### Вариант 2: Запустить скрипт индексации вручную (внутри контейнера)

```bash
docker compose exec freshmanragbot python3 -m init_scripts.index_knowledge_base
```

### Вариант 3: Локально (без Docker)

```bash
export FORCE_REINDEX=true
python3 -m init_scripts.index_knowledge_base
```

Требует настроенных переменных окружения для подключения к БД (`POSTGRES_*`) и выбранного LLM/embedding-провайдера (`LLM_PROVIDER`, `GOOGLE_API_KEY` и т.д.).

## Проверка успешности

После переиндексации проверьте логи бота:

```bash
docker compose logs -f freshmanragbot
```

Ожидаемый вывод включает что-то вроде:
```
Embedding provider: google, dimensions: 3072
Loaded N chunks total. Generating embeddings...
Successfully indexed knowledge base. N chunks added.
```

## Что индексируется

- `facts/universities/*.yaml`, `facts/language/*.yaml`, `facts/financial/*.yaml` — через `crag/yaml_facts_indexer.py` (основной источник в текущем репозитории)
- Markdown-гайды в `knowledge_base/` (если эта директория присутствует) — через `init_scripts/index_knowledge_base.py`. В текущем репозитории такой директории нет, поэтому индексируются только YAML-факты из `facts/`.

Все чанки сохраняются в таблицу `simple_documents` (pgvector) с embedding'ами, сгенерированными выбранным провайдером.

## Если что-то пошло не так

1. Проверьте логи: `docker compose logs -f freshmanragbot`
2. Убедитесь, что индексация завершилась успешно (см. "Successfully indexed")
3. Если embedding-провайдер изменился — таблица `simple_documents` будет пересоздана автоматически (см. `docs/REINDEX_KB.md`)

## Rollback (если нужно)

Если хочешь вернуться к старой версии кода:
```bash
git revert HEAD
docker compose up -d --build
```
