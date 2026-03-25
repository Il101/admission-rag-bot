"""
Agent tools for the admission bot.

Provides structured tools that the agent can invoke via function calling
to perform specific actions like checking deadlines, calculating budgets, etc.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Definition of an agent tool."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    function: Callable


# ── Tool Implementations ─────────────────────────────────────────────────


async def check_deadline(
    university: str,
    level: str,
    semester: str = "ws"
) -> dict:
    """Check admission deadlines for a specific university.

    Args:
        university: University name (e.g., "Uni Wien", "TU Wien")
        level: Study level ("bachelor", "master", "phd")
        semester: Target semester ("ws" for winter, "ss" for summer)

    Returns:
        Dictionary with deadline info or error message
    """
    # Normalize inputs
    uni_key = university.lower().replace(" ", "_").replace("-", "_")
    level_key = level.lower()
    sem_key = semester.lower()

    # Deadline database (to be expanded or moved to KB/DB)
    DEADLINES = {
        # Uni Wien
        ("uni_wien", "bachelor", "ws"): {
            "deadline": "2026-09-05",
            "non_eu_early": "2026-02-01",
            "note": "Для Non-EU: подача через u:space до 5 сентября",
        },
        ("uni_wien", "bachelor", "ss"): {
            "deadline": "2026-02-05",
            "non_eu_early": "2025-10-01",
        },
        ("uni_wien", "master", "ws"): {
            "deadline": "2026-09-05",
            "non_eu_early": "2026-02-01",
        },
        ("uni_wien", "master", "ss"): {
            "deadline": "2026-02-05",
        },
        # TU Wien
        ("tu_wien", "bachelor", "ws"): {
            "deadline": "2026-09-05",
            "non_eu_early": "2026-03-31",
            "note": "Для технических направлений — вступительные экзамены в июне",
        },
        ("tu_wien", "master", "ws"): {
            "deadline": "2026-09-05",
        },
        # WU Wien
        ("wu_wien", "bachelor", "ws"): {
            "deadline": "2026-05-31",
            "note": "Aufnahmeverfahren обязателен",
        },
        # Uni Graz
        ("uni_graz", "bachelor", "ws"): {
            "deadline": "2026-09-05",
            "non_eu_early": "2026-07-15",
        },
        ("uni_graz", "master", "ws"): {
            "deadline": "2026-09-05",
        },
        # Uni Innsbruck
        ("uni_innsbruck", "bachelor", "ws"): {
            "deadline": "2026-05-15",
            "note": "Ранний дедлайн для популярных программ",
        },
        # TU Graz
        ("tu_graz", "bachelor", "ws"): {
            "deadline": "2026-09-05",
        },
        # JKU Linz
        ("jku_linz", "bachelor", "ws"): {
            "deadline": "2026-09-05",
        },
        # MedUni Wien (special)
        ("meduni_wien", "bachelor", "ws"): {
            "deadline": "2026-03-31",
            "note": "MedAT экзамен в июле. Регистрация до 31 марта!",
        },
    }

    key = (uni_key, level_key, sem_key)
    data = DEADLINES.get(key)

    if not data:
        # Try partial match
        for (u, l, s), d in DEADLINES.items():
            if uni_key in u or u in uni_key:
                if l == level_key and s == sem_key:
                    data = d
                    break

    if data:
        deadline_date = datetime.strptime(data["deadline"], "%Y-%m-%d")
        days_left = (deadline_date - datetime.now()).days

        result = {
            "university": university,
            "level": level,
            "semester": "Зимний семестр" if sem_key == "ws" else "Летний семестр",
            "deadline": data["deadline"],
            "days_left": days_left,
            "status": "прошёл" if days_left < 0 else (
                "СРОЧНО!" if days_left < 30 else "открыт"
            ),
        }

        if data.get("non_eu_early"):
            early_date = datetime.strptime(data["non_eu_early"], "%Y-%m-%d")
            early_days = (early_date - datetime.now()).days
            result["non_eu_early_deadline"] = data["non_eu_early"]
            result["non_eu_early_days_left"] = early_days

        if data.get("note"):
            result["note"] = data["note"]

        return result

    return {
        "error": f"Дедлайн для {university} ({level}, {semester}) не найден в базе",
        "suggestion": "Уточните название вуза или проверьте официальный сайт",
    }


async def calculate_budget(
    city: str,
    months: int = 12,
    lifestyle: str = "medium"
) -> dict:
    """Calculate estimated budget for studying in Austria.

    Args:
        city: City name (Vienna, Graz, Innsbruck, Linz, Salzburg)
        months: Number of months to calculate for
        lifestyle: Spending level ("low", "medium", "high")

    Returns:
        Dictionary with budget breakdown
    """
    # Monthly costs by city (in EUR)
    COSTS = {
        "vienna": {
            "rent": 600,      # WG/Studentenheim
            "food": 300,
            "transport": 50,  # Semesterticket
            "misc": 150,
            "insurance": 65,
        },
        "wien": {
            "rent": 600,
            "food": 300,
            "transport": 50,
            "misc": 150,
            "insurance": 65,
        },
        "graz": {
            "rent": 450,
            "food": 280,
            "transport": 40,
            "misc": 120,
            "insurance": 65,
        },
        "innsbruck": {
            "rent": 550,
            "food": 290,
            "transport": 45,
            "misc": 130,
            "insurance": 65,
        },
        "linz": {
            "rent": 480,
            "food": 280,
            "transport": 40,
            "misc": 120,
            "insurance": 65,
        },
        "salzburg": {
            "rent": 550,
            "food": 290,
            "transport": 45,
            "misc": 130,
            "insurance": 65,
        },
    }

    multipliers = {
        "low": 0.75,
        "medium": 1.0,
        "high": 1.4,
    }

    city_lower = city.lower().replace("вена", "vienna").replace("грац", "graz")
    city_lower = city_lower.replace("инсбрук", "innsbruck").replace("линц", "linz")
    city_lower = city_lower.replace("зальцбург", "salzburg")

    if city_lower not in COSTS:
        return {
            "error": f"Город '{city}' не найден",
            "available_cities": ["Vienna", "Graz", "Innsbruck", "Linz", "Salzburg"],
        }

    base = COSTS[city_lower]
    mult = multipliers.get(lifestyle.lower(), 1.0)

    breakdown = {k: round(v * mult) for k, v in base.items()}
    monthly_total = sum(breakdown.values())
    yearly_total = monthly_total * 12

    return {
        "city": city,
        "lifestyle": lifestyle,
        "monthly_eur": monthly_total,
        "total_eur": round(monthly_total * months),
        "months": months,
        "breakdown": breakdown,
        "blocked_account_required": yearly_total,
        "blocked_account_note": f"Для ВНЖ нужен блокированный счёт на €{yearly_total} (12 месяцев)",
        "tips": [
            "Studentenheim дешевле, чем WG — подавайте заявку заранее",
            "Semesterticket даёт скидку на транспорт",
            "Mensa (студенческая столовая) — обед от €5-7",
        ],
    }


async def get_document_checklist(
    target_level: str,
    country: str = "RU",
    has_german_cert: bool = False,
    university: Optional[str] = None
) -> dict:
    """Generate personalized document checklist.

    Args:
        target_level: Study level ("bachelor", "master", "phd")
        country: Country of origin code (RU, UA, BY, KZ)
        has_german_cert: Whether user already has German certificate
        university: Target university (optional, for specific requirements)

    Returns:
        Dictionary with checklist items
    """
    # Base documents for all
    checklist = [
        {
            "doc": "Загранпаспорт",
            "status": "required",
            "note": "Действителен минимум 6 месяцев после подачи на визу",
        },
        {
            "doc": "Фотография 3.5x4.5 см",
            "status": "required",
            "note": "Биометрическое фото, белый фон",
        },
    ]

    # Education documents based on level
    if target_level.lower() == "bachelor":
        checklist.extend([
            {
                "doc": "Аттестат о среднем образовании",
                "status": "required",
                "note": "Оригинал + нотариально заверенная копия",
            },
            {
                "doc": "Апостиль на аттестат",
                "status": "required",
                "note": "Получить в Минобрнауки или МИД",
            },
            {
                "doc": "Приложение к аттестату с оценками",
                "status": "required",
                "note": "С апостилем",
            },
            {
                "doc": "Присяжный перевод аттестата на немецкий",
                "status": "required",
                "note": "Только сертифицированный переводчик (beglaubigte Übersetzung)",
            },
            {
                "doc": "Studienplatznachweis (для Non-EU)",
                "status": "required",
                "note": "Подтверждение права на обучение в стране происхождения",
            },
        ])
    elif target_level.lower() == "master":
        checklist.extend([
            {
                "doc": "Диплом бакалавра",
                "status": "required",
                "note": "Оригинал + нотариальная копия",
            },
            {
                "doc": "Апостиль на диплом",
                "status": "required",
            },
            {
                "doc": "Приложение к диплому (Transcript)",
                "status": "required",
                "note": "С указанием часов и оценок",
            },
            {
                "doc": "Присяжный перевод диплома",
                "status": "required",
            },
            {
                "doc": "Мотивационное письмо",
                "status": "recommended",
                "note": "Для многих Master-программ",
            },
            {
                "doc": "CV/Резюме",
                "status": "recommended",
            },
        ])
    elif target_level.lower() == "phd":
        checklist.extend([
            {
                "doc": "Диплом магистра",
                "status": "required",
            },
            {
                "doc": "Апостиль на диплом магистра",
                "status": "required",
            },
            {
                "doc": "Research proposal",
                "status": "required",
                "note": "План исследования на 3-5 страниц",
            },
            {
                "doc": "Рекомендательные письма (2-3)",
                "status": "required",
            },
        ])

    # Language certificate
    if not has_german_cert:
        checklist.append({
            "doc": "Сертификат немецкого языка (B2/C1)",
            "status": "pending",
            "note": "ÖSD, Goethe, TestDaF, DSH. Для поступления обычно B2, некоторые программы C1",
        })
    else:
        checklist.append({
            "doc": "Сертификат немецкого языка",
            "status": "completed",
        })

    # Visa/ВНЖ documents
    checklist.extend([
        {
            "doc": "Подтверждение финансов (блокированный счёт)",
            "status": "required",
            "note": f"~€12,000-14,000 на год (Sperrkonto или Verpflichtungserklärung)",
        },
        {
            "doc": "Медицинская страховка",
            "status": "required",
            "note": "ÖGK после регистрации или частная на первое время",
        },
        {
            "doc": "Справка о несудимости",
            "status": "required",
            "note": "С апостилем, переведённая",
        },
        {
            "doc": "Zulassungsbescheid (письмо о зачислении)",
            "status": "required",
            "note": "Получите после подачи документов в вуз",
        },
    ])

    # Country-specific notes
    country_notes = {
        "RU": "Апостиль: Минобрнауки (диплом/аттестат) или МИД (другие документы)",
        "UA": "Упрощённый режим для граждан Украины — уточните в посольстве",
        "BY": "Апостиль в Минюсте Беларуси",
        "KZ": "Апостиль в Минюсте Казахстана",
    }

    return {
        "level": target_level,
        "country": country,
        "checklist": checklist,
        "total_documents": len(checklist),
        "required_count": len([c for c in checklist if c["status"] == "required"]),
        "pending_count": len([c for c in checklist if c["status"] == "pending"]),
        "country_note": country_notes.get(country.upper(), ""),
        "general_tips": [
            "Все переводы должны быть присяжными (beglaubigte Übersetzung)",
            "Апостиль ставится на ОРИГИНАЛ документа",
            "Делайте нотариальные копии ПОСЛЕ апостилирования",
            "Сохраняйте электронные копии всех документов",
        ],
    }


# ── Tool Registry ────────────────────────────────────────────────────────


TOOLS = [
    Tool(
        name="check_deadline",
        description="Проверяет дедлайны подачи документов для конкретного вуза и уровня обучения",
        parameters={
            "type": "object",
            "properties": {
                "university": {
                    "type": "string",
                    "description": "Название вуза (например, 'Uni Wien', 'TU Wien', 'WU Wien')",
                },
                "level": {
                    "type": "string",
                    "enum": ["bachelor", "master", "phd"],
                    "description": "Уровень обучения",
                },
                "semester": {
                    "type": "string",
                    "enum": ["ws", "ss"],
                    "description": "Семестр: ws (зимний) или ss (летний). По умолчанию ws.",
                    "default": "ws",
                },
            },
            "required": ["university", "level"],
        },
        function=check_deadline,
    ),
    Tool(
        name="calculate_budget",
        description="Рассчитывает примерный бюджет на обучение и проживание в Австрии",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Город (Vienna/Вена, Graz/Грац, Innsbruck, Linz, Salzburg)",
                },
                "months": {
                    "type": "integer",
                    "description": "Количество месяцев для расчёта. По умолчанию 12.",
                    "default": 12,
                },
                "lifestyle": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Уровень трат: low (экономный), medium (средний), high (комфортный)",
                    "default": "medium",
                },
            },
            "required": ["city"],
        },
        function=calculate_budget,
    ),
    Tool(
        name="get_document_checklist",
        description="Генерирует персональный чек-лист документов для поступления",
        parameters={
            "type": "object",
            "properties": {
                "target_level": {
                    "type": "string",
                    "enum": ["bachelor", "master", "phd"],
                    "description": "Уровень обучения",
                },
                "country": {
                    "type": "string",
                    "description": "Код страны происхождения (RU, UA, BY, KZ)",
                    "default": "RU",
                },
                "has_german_cert": {
                    "type": "boolean",
                    "description": "Есть ли уже сертификат немецкого языка",
                    "default": False,
                },
                "university": {
                    "type": "string",
                    "description": "Целевой вуз (опционально, для специфичных требований)",
                },
            },
            "required": ["target_level"],
        },
        function=get_document_checklist,
    ),
]


def get_tool_schemas() -> list[dict]:
    """Get JSON schemas for all tools (for LLM function calling)."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in TOOLS
    ]


def get_tool_by_name(name: str) -> Optional[Tool]:
    """Get a tool by its name."""
    for tool in TOOLS:
        if tool.name == name:
            return tool
    return None


async def execute_tool(name: str, arguments: dict) -> dict:
    """Execute a tool by name with given arguments."""
    tool = get_tool_by_name(name)
    if not tool:
        return {"error": f"Tool '{name}' not found"}

    try:
        result = await tool.function(**arguments)
        return result
    except Exception as e:
        logger.error(f"Tool '{name}' execution failed: {e}")
        return {"error": str(e)}
