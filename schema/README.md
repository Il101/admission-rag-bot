# Schema для structured facts

Этот документ описывает структуру YAML файлов в директории `facts/`.

## Принципы

1. **Один файл = одна сущность** (университет, программа VWU, etc.)
2. **Только факты** — никаких объяснений, советов, мнений
3. **Обязательные поля**: `last_verified`, `source_url`
4. **Даты в ISO формате**: `2026-07-15`
5. **Деньги**: `amount` + `currency` + `period`

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

### Process Facts (`facts/processes/*.yaml`)

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
