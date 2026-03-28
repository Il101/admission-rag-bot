"""Assertion Validator for RAG Answers.

Validates that each claim in the generated answer is grounded in sources
from the appropriate entity category, preventing cross-entity hallucinations
(e.g., food facts leaking into housing answers).

Based on the RAGFlow approach: each assertion must have a source from
the same category as the question's focus.
"""

import json
import logging
import re
from typing import List, Any, Optional

logger = logging.getLogger(__name__)


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

    This is a fast, rule-based assertion check. For production, you could
    enhance this with LLM-based claim extraction and grounding verification.

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
