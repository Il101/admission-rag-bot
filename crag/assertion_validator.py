"""Assertion / Faithfulness Validator for RAG Answers.

Primary path: ``verify_faithfulness`` — an LLM-judge faithfulness gate that
checks whether factual claims in the generated answer are supported by the
retrieved context (``docs``). This replaces the old keyword/category-based
validator as the primary check used by ``AssertionCheckStep``.

The legacy lightweight checks (``validate_answer_assertions``,
``_check_source_category_match``, ``_detect_answer_topic``,
``_extract_entity_types_from_docs``, ``add_assertion_disclaimer``) are kept
for backward compatibility / tests but are demoted — no longer the primary
grounding check in the pipeline.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Any, Optional

logger = logging.getLogger(__name__)


# ── Faithfulness gate (LLM judge) ────────────────────────────────────────


@dataclass
class FaithfulnessResult:
    """Result of an LLM-judge faithfulness check.

    Attributes:
        is_faithful: True if the answer's factual claims are supported by
            the provided context (or if the check could not be completed —
            fail-open).
        unsupported_claims: Concrete factual statements from the answer that
            the judge could not find support for in the context. Empty when
            ``is_faithful`` is True.
        reason: Short human-readable explanation (for logging).
    """

    is_faithful: bool
    unsupported_claims: List[str] = field(default_factory=list)
    reason: str = ""


def _build_faithfulness_context(docs: List[Any]) -> str:
    """Build a compact CONTEXT block from retrieved/reranked docs for the judge."""
    parts = []
    for doc in docs:
        content = getattr(doc, "page_content", "") or ""
        if content:
            parts.append(content)
    return "\n\n".join(parts)


_FAITHFULNESS_PROMPT_TEMPLATE = """Ты — строгий проверяющий фактологическую точность ответов ассистента по поступлению в Австрию.

Тебе даны:
1. КОНТЕКСТ — факты из базы знаний, на основе которых ассистент должен был отвечать.
2. ВОПРОС пользователя.
3. ОТВЕТ ассистента.

ЗАДАЧА: Найди в ОТВЕТЕ конкретные фактические утверждения (даты, цифры/суммы, названия учреждений/документов, требования, сроки, адреса), которые НЕ подтверждаются КОНТЕКСТОМ — то есть ассистент их выдумал или взял не из контекста.

ИГНОРИРУЙ:
- Приветствия, общие фразы, эмпатийные вставки.
- Общие советы без конкретики (например, "рекомендуем уточнить детали у вуза").
- Блок со списком источников ("Источники:", "[1] ...").
- Предложенные вопросы (suggested_questions).
- Перефразирование/обобщение фактов, которые ЕСТЬ в контексте (даже если формулировка отличается).

БУДЬ КОНСЕРВАТИВЕН: помечай как неподтверждённые ТОЛЬКО явные фактические утверждения (конкретные даты, числа, названия, требования), для которых в контексте действительно нет основания. Если сомневаешься — не помечай.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:
{answer}

Ответь СТРОГО в формате JSON, без пояснений вне JSON:
{{"is_faithful": true/false, "unsupported_claims": ["...", "..."]}}

Если все утверждения подтверждены (или ответ не содержит конкретных фактов) — {{"is_faithful": true, "unsupported_claims": []}}."""


def _parse_faithfulness_response(response_text: str) -> "FaithfulnessResult":
    """Parse the judge LLM's JSON response into a FaithfulnessResult.

    Raises on parse failure so the caller can fail-open.
    """
    raw = (response_text or "").strip()

    # Tolerate responses wrapped in markdown code fences or extra prose.
    if "{" in raw and "}" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        raw = raw[start:end]

    data = json.loads(raw)

    is_faithful = bool(data.get("is_faithful", True))
    unsupported = data.get("unsupported_claims", []) or []
    if not isinstance(unsupported, list):
        unsupported = [str(unsupported)]
    unsupported = [str(c) for c in unsupported]

    # Consistency guard: if claims were listed but is_faithful wasn't
    # explicitly set to False, treat non-empty claims as "not faithful".
    if unsupported and is_faithful:
        is_faithful = False

    reason = ""
    if not is_faithful:
        reason = f"{len(unsupported)} unsupported claim(s)"

    return FaithfulnessResult(
        is_faithful=is_faithful,
        unsupported_claims=unsupported,
        reason=reason,
    )


async def verify_faithfulness(
    answer_text: str,
    docs: List[Any],
    question: str,
    llm_provider: Optional[Any],
) -> "FaithfulnessResult":
    """Check whether ``answer_text`` is faithful to ``docs`` using an LLM judge.

    Calls ``llm_provider.generate(prompt=..., system_prompt=None,
    temperature=0)`` with a judge prompt that asks for a JSON verdict:
    ``{"is_faithful": bool, "unsupported_claims": [...]}``.

    Fail-open: on any LLM/parse error (or if ``llm_provider`` is None),
    returns ``is_faithful=True`` so infrastructure issues never block an
    answer from reaching the user.

    Args:
        answer_text: The generated answer text (plain text, may include
            HTML formatting and ``[n]`` citation markers).
        docs: The documents (context) the answer should be grounded in.
        question: The original user question (for judge context).
        llm_provider: An object exposing an async ``generate(prompt=...,
            system_prompt=..., temperature=...)`` method.

    Returns:
        FaithfulnessResult
    """
    if llm_provider is None:
        return FaithfulnessResult(is_faithful=True, reason="no llm_provider (fail-open)")

    context = _build_faithfulness_context(docs)
    if not context.strip():
        # Nothing to check against — fail open rather than flag everything.
        return FaithfulnessResult(is_faithful=True, reason="empty context (fail-open)")

    prompt = _FAITHFULNESS_PROMPT_TEMPLATE.format(
        context=context[:12000],  # bound prompt size
        question=question or "",
        answer=answer_text[:6000],
    )

    try:
        response_text = await llm_provider.generate(
            prompt=prompt,
            system_prompt=None,
            temperature=0.0,
        )
        return _parse_faithfulness_response(response_text)
    except Exception as e:
        logger.warning(f"[Faithfulness] LLM judge failed, failing open: {e}")
        return FaithfulnessResult(is_faithful=True, reason=f"judge error (fail-open): {e}")


# ── Legacy keyword/category checks (demoted, kept for compatibility) ────


class AssertionCheckResult:
    """Result of assertion validation."""

    def __init__(
        self,
        is_valid: bool,
        warnings: List[str],
        filtered_answer: Optional[str] = None,
    ):
        self.is_valid = is_valid
        self.warnings = warnings
        self.filtered_answer = filtered_answer


def _extract_entity_types_from_docs(docs: List[Any]) -> set[str]:
    """Extract all entity types from retrieved documents."""
    entity_types = set()
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        entity_type = metadata.get("entity_type", "general")
        entity_types.add(entity_type)
    return entity_types


def _detect_answer_topic(answer: str) -> Optional[str]:
    """Detect the main topic of the answer based on keywords."""
    answer_lower = answer.lower()

    # Topic keyword mapping (using word boundaries to avoid false positives)
    topic_keywords = {
        "housing": ["жиль", "общежит", "studentenheim", "квартир", "аренд", "wohnung", "miet", "meldezettel"],
        "food": ["питан", "еда", "столов", "mensa", "кафе", "ресторан", "меню", "cafeteria", "meal"],
        "visa": ["виз", "visa", "residence permit", "вид на жительство", "aufenthaltstitel"],
        "finance": ["стоимость", "цена", "цены", "tuition", "studienbeitrag", "оплат", "fee", "scholarship", "евро за"],
        "language": ["язык", "german", "английск", "немецк", "language", "сертифик", "certificate"],
        "admission": ["поступ", "прием", "дедлайн", "deadline", "application", "заявк", "зачисл"],
    }

    # Count keyword matches per topic
    topic_scores = {}
    for topic, keywords in topic_keywords.items():
        count = sum(1 for kw in keywords if kw in answer_lower)
        if count > 0:
            topic_scores[topic] = count

    if not topic_scores:
        return None

    # Return topic with highest score
    return max(topic_scores.items(), key=lambda x: x[1])[0]


def _check_source_category_match(
    answer: str,
    docs: List[Any],
    question: str,
) -> AssertionCheckResult:
    """Check if answer's topic matches sources' entity types.

    This is a lightweight assertion check that doesn't require LLM calls.
    It detects if the answer discusses a topic (e.g., housing) but sources
    are from a different category (e.g., food).

    Args:
        answer: Generated answer text
        docs: Retrieved/reranked documents used for answer
        question: Original user question

    Returns:
        AssertionCheckResult with validation status and warnings
    """
    if not docs:
        return AssertionCheckResult(
            is_valid=True,
            warnings=["No sources available for assertion check"],
        )

    # Extract entity types from sources
    source_entity_types = _extract_entity_types_from_docs(docs)

    # Detect main topic of the answer
    answer_topic = _detect_answer_topic(answer)

    if not answer_topic:
        # Answer is too generic to validate
        return AssertionCheckResult(
            is_valid=True,
            warnings=[],
        )

    # Check if sources match answer topic
    # Allow "general" category to be valid for any topic
    if answer_topic in source_entity_types or "general" in source_entity_types:
        return AssertionCheckResult(
            is_valid=True,
            warnings=[],
        )

    # Category mismatch detected
    warning = (
        f"Answer discusses {answer_topic} but sources are from: "
        f"{', '.join(sorted(source_entity_types))}"
    )

    logger.warning(f"[Assertion Check] Category mismatch: {warning}")

    return AssertionCheckResult(
        is_valid=False,
        warnings=[warning],
        filtered_answer=None,  # Could add filtering logic here
    )


async def validate_answer_assertions(
    answer: str,
    docs: List[Any],
    question: str,
    llm_provider: Optional[Any] = None,
) -> AssertionCheckResult:
    """Validate that answer assertions are grounded in appropriate sources.

    DEMOTED: this fast, rule-based category-matching check is no longer the
    primary grounding check in the pipeline — ``AssertionCheckStep`` now uses
    :func:`verify_faithfulness` (LLM judge) instead. This function is kept
    for backward compatibility and is still covered by tests.

    Args:
        answer: Generated answer text
        docs: Retrieved/reranked documents
        question: Original question
        llm_provider: Optional LLM for advanced claim extraction

    Returns:
        AssertionCheckResult with validation results
    """
    # For now, use lightweight category matching
    # Can be extended with LLM-based claim extraction if needed
    result = _check_source_category_match(answer, docs, question)

    if not result.is_valid:
        logger.info(
            f"[Assertion Check] Failed validation: {'; '.join(result.warnings)}"
        )

    return result


def add_assertion_disclaimer(
    answer: str,
    warnings: List[str],
) -> str:
    """Add disclaimer to answer if assertion check found issues.

    Args:
        answer: Original answer text
        warnings: List of warning messages from assertion check

    Returns:
        Answer with disclaimer prepended
    """
    if not warnings:
        return answer

    disclaimer = (
        "⚠️ Внимание: Ответ может содержать информацию из разных категорий. "
        "Пожалуйста, проверьте источники.\n\n"
    )

    return disclaimer + answer
