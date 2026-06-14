"""
Agent tools for the admission bot.

Provides structured tools that the agent can invoke via function calling
to perform specific actions like checking deadlines, calculating budgets, etc.

Tools are categorized into:
1. Personal Progress Tools - work with user's journey state (no RAG needed)
2. Calculator Tools - quick computations (no RAG needed)
3. Knowledge Base Tools - complex queries (RAG required)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Any, Optional, List

import yaml

logger = logging.getLogger(__name__)


# Path to the cost-of-living YAML, resolved relative to this file's location
# (crag/tools.py -> repo root is the parent directory).
COST_OF_LIVING_PATH = Path(__file__).resolve().parent.parent / "facts" / "financial" / "cost-of-living.yaml"


@lru_cache(maxsize=1)
def _load_cost_of_living() -> Optional[dict]:
    """Load and cache the cost-of-living data from the facts YAML.

    Returns:
        Parsed YAML data as a dict, or None if the file is missing/invalid.
    """
    try:
        with open(COST_OF_LIVING_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        logger.error(f"Cost-of-living data file not found: {COST_OF_LIVING_PATH}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse cost-of-living YAML: {e}")
        return None


@dataclass
class Tool:
    """Definition of an agent tool."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    function: Callable


# ── Tool Implementations ─────────────────────────────────────────────────


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
    data = _load_cost_of_living()

    if not data:
        return {
            "error": "Данные о стоимости жизни не найдены (facts/financial/cost-of-living.yaml)",
        }

    costs = data.get("cities", {})

    multipliers = {
        "low": 0.75,
        "medium": 1.0,
        "high": 1.4,
    }

    city_lower = city.lower().replace("вена", "vienna").replace("грац", "graz")
    city_lower = city_lower.replace("инсбрук", "innsbruck").replace("линц", "linz")
    city_lower = city_lower.replace("зальцбург", "salzburg")

    if city_lower not in costs:
        return {
            "error": f"Город '{city}' не найден",
            "available_cities": ["Vienna", "Graz", "Innsbruck", "Linz", "Salzburg"],
        }

    base = costs[city_lower]
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
        "tips": data.get("tips", []),
    }


# ── Personal Progress Tools ──────────────────────────────────────────


async def get_my_progress(
    session_factory,
    tg_id: int,
) -> dict:
    """Get user's current progress through admission journey.

    Args:
        session_factory: Database session factory
        tg_id: Telegram user ID

    Returns:
        Dictionary with journey stages and completion status
    """
    from bot.db import get_user_memory
    from bot.memory import JOURNEY_STAGES

    try:
        async with session_factory() as session:
            memory = await get_user_memory(session, tg_id)
            journey_state = memory.get("journey_state") or {}

            if not journey_state:
                return {
                    "status": "not_started",
                    "message": "Ты ещё не начал процесс поступления. Давай начнём!",
                    "total_stages": len(JOURNEY_STAGES),
                    "completed_stages": 0,
                }

            stages = []
            completed_count = 0

            for stage_id, stage_info in JOURNEY_STAGES.items():
                status = journey_state.get(stage_id, "pending")
                if status == "discussed":
                    completed_count += 1

                stages.append({
                    "id": stage_id,
                    "label": stage_info["label"],
                    "status": status,
                    "completed": status == "discussed",
                })

            progress_pct = int((completed_count / len(JOURNEY_STAGES)) * 100)

            return {
                "status": "in_progress",
                "progress_percent": progress_pct,
                "completed_stages": completed_count,
                "total_stages": len(JOURNEY_STAGES),
                "stages": stages,
                "message": f"Ты прошёл {completed_count} из {len(JOURNEY_STAGES)} этапов ({progress_pct}%)",
            }

    except Exception as e:
        logger.error(f"get_my_progress failed: {e}")
        return {"error": str(e)}


async def get_next_steps(
    session_factory,
    tg_id: int,
) -> dict:
    """Get actionable next steps based on user's current progress.

    Args:
        session_factory: Database session factory
        tg_id: Telegram user ID

    Returns:
        Dictionary with recommended next actions
    """
    from bot.db import get_user_memory, get_user
    from bot.memory import JOURNEY_STAGES

    try:
        async with session_factory() as session:
            user = await get_user(session, tg_id)
            memory = await get_user_memory(session, tg_id)
            journey_state = memory.get("journey_state") or {}

            if not user or not journey_state:
                return {
                    "next_steps": [
                        "📋 Пройди опрос для анализа твоей ситуации",
                        "🎯 Определись с целевым уровнем обучения (бакалавр/магистр)",
                        "🏫 Выбери интересующие университеты",
                    ],
                    "priority": "high",
                    "message": "Начни с базовых шагов",
                }

            next_steps = []

            # Check pending stages with completed prerequisites
            for stage_id, stage_info in JOURNEY_STAGES.items():
                if journey_state.get(stage_id) == "discussed":
                    continue

                # Check if prerequisites are met
                prereqs = stage_info.get("prerequisites", [])
                prereqs_met = all(
                    journey_state.get(p) == "discussed" for p in prereqs
                )

                if not prereqs or prereqs_met:
                    next_steps.append({
                        "stage": stage_info["label"],
                        "action": stage_info["question"],
                        "priority": "high" if not prereqs else "medium",
                    })

            # Add specific deadline warnings if user has mentioned universities
            from bot.db import get_user_entities
            entities = await get_user_entities(session, tg_id, entity_type="university")

            if entities:
                next_steps.insert(0, {
                    "stage": "⏰ Дедлайны",
                    "action": f"Проверь дедлайны для {entities[0]['value']}",
                    "priority": "urgent",
                })

            return {
                "next_steps": next_steps[:5],  # Top 5 actions
                "total_pending": len([s for s in journey_state.values() if s != "discussed"]),
                "message": "Вот что тебе стоит сделать дальше",
            }

    except Exception as e:
        logger.error(f"get_next_steps failed: {e}")
        return {"error": str(e)}


async def get_my_profile(
    session_factory,
    tg_id: int,
) -> dict:
    """Get user's profile and preferences.

    Args:
        session_factory: Database session factory
        tg_id: Telegram user ID

    Returns:
        Dictionary with user profile data
    """
    from bot.db import get_user, get_user_entities

    try:
        async with session_factory() as session:
            user = await get_user(session, tg_id)

            if not user:
                return {
                    "status": "not_found",
                    "message": "Профиль не найден. Начни с команды /start",
                }

            # Get entities (universities, cities, etc.)
            entities = await get_user_entities(session, tg_id)

            profile = {
                "country": user.country,
                "document_type": user.document_type,
                "target_level": user.target_level,
                "german_level": user.german_level,
                "english_level": user.english_level,
                "created_at": user.created_at.strftime("%Y-%m-%d") if user.created_at else None,
                "last_active": user.last_active.strftime("%Y-%m-%d") if user.last_active else None,
            }

            # Group entities by type
            universities = [e["value"] for e in entities if e["type"] == "university"]
            cities = [e["value"] for e in entities if e["type"] == "city"]
            programs = [e["value"] for e in entities if e["type"] == "program"]

            return {
                "profile": profile,
                "interests": {
                    "universities": universities,
                    "cities": cities,
                    "programs": programs,
                },
                "message": "Твой профиль абитуриента",
            }

    except Exception as e:
        logger.error(f"get_my_profile failed: {e}")
        return {"error": str(e)}


async def get_my_entities(
    session_factory,
    tg_id: int,
    entity_type: Optional[str] = None,
) -> dict:
    """Get user's tracked entities (universities, deadlines, documents, etc.).

    Args:
        session_factory: Database session factory
        tg_id: Telegram user ID
        entity_type: Optional filter by entity type

    Returns:
        Dictionary with user's entities
    """
    from bot.db import get_user_entities

    try:
        async with session_factory() as session:
            entities = await get_user_entities(session, tg_id, entity_type)

            if not entities:
                return {
                    "entities": [],
                    "count": 0,
                    "message": "У тебя пока нет сохранённых данных" + (
                        f" типа '{entity_type}'" if entity_type else ""
                    ),
                }

            # Group by type
            by_type = {}
            for e in entities:
                t = e["type"]
                if t not in by_type:
                    by_type[t] = []
                by_type[t].append(e["value"])

            return {
                "entities": entities,
                "by_type": by_type,
                "count": len(entities),
                "message": f"Найдено {len(entities)} записей",
            }

    except Exception as e:
        logger.error(f"get_my_entities failed: {e}")
        return {"error": str(e)}


async def calculate_days_until(
    target_date: str,
) -> dict:
    """Calculate days remaining until a specific date.

    Args:
        target_date: Date in format YYYY-MM-DD

    Returns:
        Dictionary with days remaining and urgency level
    """
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d")
        now = datetime.now()
        days_left = (target - now).days

        if days_left < 0:
            urgency = "expired"
            message = f"Дедлайн прошёл {abs(days_left)} дней назад"
        elif days_left == 0:
            urgency = "today"
            message = "Дедлайн СЕГОДНЯ!"
        elif days_left <= 7:
            urgency = "urgent"
            message = f"Срочно! Осталось всего {days_left} дней"
        elif days_left <= 30:
            urgency = "soon"
            message = f"Осталось {days_left} дней"
        else:
            urgency = "ok"
            message = f"Осталось {days_left} дней"

        return {
            "target_date": target_date,
            "days_left": days_left,
            "urgency": urgency,
            "message": message,
        }

    except ValueError as e:
        return {
            "error": f"Неверный формат даты. Используй YYYY-MM-DD. Ошибка: {e}"
        }


# ── Tool Registry ────────────────────────────────────────────────────────


# Tools that require user context (session_factory + tg_id injection)
PERSONAL_TOOLS = {"get_my_progress", "get_next_steps", "get_my_profile", "get_my_entities"}

TOOLS = [
    # ── Calculator Tools (no DB access needed) ──
    Tool(
        name="calculate_budget",
        description="Рассчитывает примерный бюджет на обучение и проживание в Австрии. Используй когда пользователь спрашивает о стоимости, расходах, бюджете.",
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
        name="calculate_days_until",
        description="Вычисляет количество дней до указанной даты. Используй для проверки срочности дедлайнов.",
        parameters={
            "type": "object",
            "properties": {
                "target_date": {
                    "type": "string",
                    "description": "Целевая дата в формате YYYY-MM-DD",
                },
            },
            "required": ["target_date"],
        },
        function=calculate_days_until,
    ),

    # ── Personal Progress Tools (require tg_id injection) ──
    Tool(
        name="get_my_progress",
        description="Показывает ПЕРСОНАЛЬНЫЙ прогресс пользователя по этапам поступления. Используй когда пользователь спрашивает 'на каком я этапе', 'мой прогресс', 'что я уже сделал', 'мой чек-лист'.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        function=get_my_progress,
    ),
    Tool(
        name="get_next_steps",
        description="Даёт ПЕРСОНАЛЬНЫЕ рекомендации что делать дальше. Используй когда пользователь спрашивает 'что дальше', 'следующие шаги', 'что мне делать', 'с чего начать'.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        function=get_next_steps,
    ),
    Tool(
        name="get_my_profile",
        description="Возвращает профиль пользователя: страна, цель, университеты. Используй когда пользователь спрашивает 'мой профиль', 'моя информация', 'что ты знаешь обо мне'.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        function=get_my_profile,
    ),
    Tool(
        name="get_my_entities",
        description="Возвращает сохранённые сущности пользователя (университеты, города, программы). Используй когда нужно узнать предпочтения пользователя.",
        parameters={
            "type": "object",
            "properties": {
                "entity_type": {
                    "type": "string",
                    "enum": ["university", "city", "program", "deadline", "document"],
                    "description": "Фильтр по типу сущности (опционально)",
                },
            },
            "required": [],
        },
        function=get_my_entities,
    ),
]


def get_tool_schemas(include_personal: bool = True) -> list[dict]:
    """Get JSON schemas for tools (for LLM function calling).

    Args:
        include_personal: Include personal tools that need tg_id injection
    """
    schemas = []
    for tool in TOOLS:
        if not include_personal and tool.name in PERSONAL_TOOLS:
            continue
        # For personal tools, don't expose internal params
        if tool.name in PERSONAL_TOOLS:
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.parameters.get("properties", {}),
                    "required": [],
                },
            })
        else:
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
    return schemas


def get_tool_by_name(name: str) -> Optional[Tool]:
    """Get a tool by its name."""
    for tool in TOOLS:
        if tool.name == name:
            return tool
    return None


def is_personal_tool(name: str) -> bool:
    """Check if tool requires user context injection."""
    return name in PERSONAL_TOOLS


async def execute_tool(
    name: str,
    arguments: dict,
    session_factory=None,
    tg_id: int = None
) -> dict:
    """Execute a tool by name with given arguments.

    Args:
        name: Tool name
        arguments: Tool arguments from LLM
        session_factory: DB session factory (for personal tools)
        tg_id: Telegram user ID (for personal tools)
    """
    tool = get_tool_by_name(name)
    if not tool:
        return {"error": f"Tool '{name}' not found"}

    try:
        # Inject user context for personal tools
        if name in PERSONAL_TOOLS:
            if not session_factory or not tg_id:
                return {"error": f"Tool '{name}' requires user context"}
            arguments["session_factory"] = session_factory
            arguments["tg_id"] = tg_id

        result = await tool.function(**arguments)
        return result
    except Exception as e:
        logger.error(f"Tool '{name}' execution failed: {e}")
        return {"error": str(e)}
