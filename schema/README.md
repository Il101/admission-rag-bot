# Schema для structured facts

Этот документ описывает структуру YAML файлов в директории `facts/`.

## Принципы

1. **Один файл = одна сущность** (университет, программа VWU, etc.)
2. **Только факты** — никаких объяснений, советов, мнений
3. **Обязательные поля**: `last_verified`, `source_url`
4. **Даты в ISO формате**: `2026-07-15`
5. **Деньги**: `amount` + `currency` + `period`

## Validity / Source Attribution (`valid_for`, `source_url`)

Для **time-sensitive фактов** — дедлайнов подачи документов, стоимости обучения
(`tuition`), а также правил по визам/ВНЖ — каждая запись должна по возможности
нести два поля:

- **`source_url`** — официальная страница, откуда взят факт. Это поле уже
  читается индексером (`crag/yaml_facts_indexer.py`) и попадает в metadata
  каждого чанка как `source_url`. Это основной источник для цитирования: бот
  должен ссылаться на него, когда сообщает дедлайн/стоимость/визовое правило.
  **Никогда не придумывайте `source_url`** — указывайте только реально
  существующую официальную страницу.

- **`valid_for`** — учебный год, к которому относится факт, в формате
  `"2026/27"` (нотация Австрии: зимний семестр года N + летний семестр года
  N+1). Это поле:
  - можно указать явно на записи (`valid_for: "2026/27"`);
  - либо индексер выводит его автоматически из поля `semester`
    (например, `WS2026/27` → `2026/27`, `SS2027` → `2026/27`) для записей
    `deadlines`, если явное значение не указано.

  Индексер кладёт `valid_for` в `metadata` чанка и добавляет в текст чанка
  короткую пометку `(актуально для приёма {valid_for})`, чтобы LLM явно видел
  период действия факта. **Никогда не указывайте `valid_for`, если год не
  выводится из имеющихся данных** (например, из `semester` или из дат самого
  дедлайна) — не угадывайте и не экстраполируйте.

Пример для дедлайна (год выводится из `semester` автоматически, поле
`valid_for` можно не указывать):

```yaml
deadlines:
  - semester: WS2026/27
    category: non-eu
    level: bachelor
    submission:
      start: 2026-01-16
      end: 2026-07-15
    source_url: https://...
    last_verified: 2026-03-01
    # valid_for: "2026/27"  — необязательно, индексер выведет из semester
```

Пример для `tuition` (нет поля `semester`, поэтому `valid_for` указывается
явно, если год определён по контексту файла — например, тот же `last_verified`
и те же дедлайны WS2026/27, что и у остальных записей этого вуза):

```yaml
tuition:
  - category: non-eu
    amount: 726.72
    currency: EUR
    period: semester
    total: 751.92
    source_url: https://...
    valid_for: "2026/27"
    last_verified: 2026-03-01
```

## Типы фактов

### University Facts (`facts/universities/*.yaml`)

```yaml
id: tu-wien                    # уникальный ID, используется в ссылках
name:
  de: Technische Universität Wien
  en: Vienna University of Technology
  ru: Венский технический университет

city: Wien
website: https://www.tuwien.at
admission_portal: https://tiss.tuwien.ac.at

# Дедлайны подачи документов
deadlines:
  - semester: WS2026/27
    category: non-eu           # non-eu | eu | all
    level: bachelor            # bachelor | master | phd
    submission:
      start: 2026-01-16
      end: 2026-07-15
    enrollment:
      start: 2026-07-01
      end: 2026-10-31
    notes: "Документы должны быть ПОЛУЧЕНЫ до дедлайна"
    source_url: https://...
    last_verified: 2026-03-01

# Требования к языку
language_requirements:
  german:
    direct_admission: C1       # уровень для прямого зачисления
    vwu_admission: A2          # уровень для зачисления на VWU
    certificates:              # принимаемые сертификаты
      - name: ÖSD Zertifikat C1
        level: C1
      - name: Goethe-Zertifikat C1
        level: C1
      - name: TestDaF
        level: TDN4
        notes: "TDN4 во всех 4 частях"
    source_url: https://...
    last_verified: 2026-03-01
  english:
    required_for: [master-english-programs]
    min_level: B2
    certificates:
      - name: TOEFL iBT
        min_score: 87
      - name: IELTS Academic
        min_score: 6.5
    source_url: https://...
    last_verified: 2026-03-01

# Стоимость обучения
tuition:
  - category: non-eu
    amount: 726.72
    currency: EUR
    period: semester
    additional:
      - name: ÖH-Beitrag
        amount: 25.20
        currency: EUR
    total: 751.92
    source_url: https://...
    valid_for: "2026/27"        # учебный год, к которому относится сумма
    last_verified: 2026-03-01
  - category: eu
    amount: 0
    currency: EUR
    period: semester
    additional:
      - name: ÖH-Beitrag
        amount: 25.20
        currency: EUR
    total: 25.20
    source_url: https://...
    valid_for: "2026/27"
    last_verified: 2026-03-01

# Вступительные экзамены (если есть)
entrance_exams:
  - program: Informatik
    places: 670
    shared_with: [Computer Engineering, Business Informatics]
    registration:
      start: 2026-04-01
      end: 2026-05-04
    exam_date: 2026-07-13
    cost: 50
    currency: EUR
    source_url: https://...
    last_verified: 2026-03-01

# Контакты
contacts:
  admission_office:
    address: "Karlsplatz 13, 1040 Wien"
    email: admission@tuwien.ac.at
    phone: "+43 1 58801 41188"
    hours: "Пн, Вт, Чт 09:30–12:00; Ср 13:30–16:00"
```

### Language Facts (`facts/language/*.yaml`)

```yaml
# Пример: vwu.yaml
id: vwu-overview

# Разные названия в разных городах
providers:
  - city: Wien
    name: VWU
    full_name: Vorstudienlehrgang der Wiener Universitäten
    website: https://vorstudienlehrgang.oead.at
    cost:
      amount: 1800
      currency: EUR
      period: semester
    last_verified: 2026-03-01

  - city: Graz
    name: VGUH
    full_name: Vorstudienlehrgang der Grazer Universitäten
    website: https://vorstudienlehrgang.oead.at/en/graz
    cost:
      amount_min: 529
      amount_max: 829
      currency: EUR
      period: semester
    last_verified: 2026-03-01

# Общие правила
rules:
  - id: vwu-lock
    title: "VWU Lock Trap"
    description: "После зачисления на VWU нельзя сдать внешний C1 сертификат — только Ergänzungsprüfung"
    applies_to: [tu-wien]
    source_url: https://...
    last_verified: 2026-03-01

# Экзамен
exam:
  name: Ergänzungsprüfung Deutsch
  level: C1
  cost:
    amount: 66
    currency: EUR
  attempts_per_semester: 3
  source_url: https://...
  last_verified: 2026-03-01
```

### Financial Facts (`facts/financial/*.yaml`)

```yaml
# Пример: student-budget.yaml
id: student-budget

cities:
  - city: Wien
    monthly_budget:
      min: 1100
      max: 1500
      currency: EUR
    breakdown:
      - category: rent
        min: 500
        max: 800
        notes: "WG или студенческое общежитие"
      - category: food
        min: 250
        max: 350
      - category: transport
        amount: 75
        notes: "Semester ticket"
      - category: health_insurance
        amount: 65
        notes: "ÖGK студенческая"
      - category: misc
        min: 100
        max: 200
    source_url: https://...
    last_verified: 2026-03-01
```

### Process Facts (`facts/processes/*.yaml`) — schema reserved, not yet populated

```yaml
# Пример: visa-d.yaml
id: visa-d-russia

country: RU
visa_type: D

requirements:
  - document: Zulassungsbescheid
    required: true
  - document: Паспорт
    required: true
    validity: "6 месяцев после окончания визы"
  - document: Финансовое обеспечение
    required: true
    amount: 13000
    currency: EUR
    period: year

processing_time:
  min_weeks: 6
  max_weeks: 12

embassy:
  city: Москва
  address: "..."
  appointment_url: https://...

source_url: https://...
last_verified: 2026-03-01
```

## Валидация (будущее)

В будущем можно добавить автоматическую валидацию:
- Даты не в прошлом
- Ссылки работают
- Обязательные поля заполнены
- `last_verified` не старше N месяцев
- Для `deadlines`/`tuition`/visa-записей: `source_url` указан и `valid_for`
  определён (явно или через `semester`)
