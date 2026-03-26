# Инструкция по переиндексации после архитектурных изменений

## Что было сделано

1. ✅ **Исправлен chunking** — таблицы теперь не режутся, overlap увеличен с 50 до 200 char
2. ✅ **Созданы atomic facts** — 4 критичных факта в `knowledge_base/atoms/`:
   - `vwu-names-by-city.md` — VWU только Вена, VGUH Грац и т.д.
   - `vwu-lock-trap.md` — нельзя бросить VWU и сдать внешний C1
   - `tu-wien-c1-requirement.md` — TU Wien требует C1, а не B2
   - `path-from-a2-to-admission.md` — путь поступления с A2
3. ✅ **Удален дубликат** — `low-german-level-path.md` (заменен на atomic facts)
4. ✅ **Увеличен top_k** — с 6 до 10 для лучшего retrieval
5. ✅ **Добавлена валидация** — bot не выдаст троеточие вместо ответа

## Как переиндексировать на Railway

### Вариант 1: Триггер через переменную окружения (рекомендуется)

```bash
# Установи Railway CLI (если еще не установлен)
brew install railway

# Войди в проект
railway link

# Триггер переиндексации
railway variables set REINDEX_TRIGGER=$(date +%s)
```

Это автоматически запустит redeploy → indexing script запустится → база обновится.

### Вариант 2: Через UI Railway

1. Зайди на railway.app
2. Открой свой проект
3. Зайди в Variables
4. Добавь новую переменную `REINDEX_TRIGGER` со значением `1`
5. Сохрани → автоматический redeploy
6. После успешной переиндексации удали эту переменную

### Вариант 3: Force reindex

```bash
railway variables set FORCE_REINDEX=true
railway up --detach
```

После успешной переиндексации:
```bash
railway variables delete FORCE_REINDEX
```

## Проверка успешности

После деплоя проверь логи:

```bash
railway logs
```

Должно быть:
```
✅ Embedding provider: nvidia, dimensions: 1536
Loaded 450 chunks total. Generating embeddings...
📄 vwu-names-by-city.md: 1 chunks
📄 vwu-lock-trap.md: 1 chunks
📄 tu-wien-c1-requirement.md: 1 chunks
📄 path-from-a2-to-admission.md: 1 chunks
Successfully indexed knowledge base. 450 chunks added.
```

## Ожидаемые улучшения

После переиндексации:
- ✅ Бот найдет факты про VWU/VGUH при вопросах о низком уровне языка
- ✅ Не будет путать VWU (Вена) с курсами в других городах
- ✅ Таблицы в ответах будут целыми
- ✅ Не будет показывать "..." вместо ответа
- ✅ Лучший retrieval благодаря увеличенному top_k

## Если что-то пошло не так

Если после переиндексации бот работает хуже:
1. Проверь логи: `railway logs`
2. Убедись что indexing завершился успешно (см. "Successfully indexed")
3. Если embeddings provider поменялся — это нормально, просто занимает 2-3 минуты

## Rollback (если нужно)

Если хочешь вернуться к старой версии:
```bash
git revert HEAD
git push
```
Railway автоматически задеплоит предыдущую версию.
