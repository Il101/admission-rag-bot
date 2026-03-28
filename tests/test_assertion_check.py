"""Tests for assertion validation system.

Validates that the assertion check correctly identifies category mismatches
and prevents cross-entity hallucinations.
"""

import pytest
from crag.assertion_validator import (
    validate_answer_assertions,
    _detect_answer_topic,
    _extract_entity_types_from_docs,
    _check_source_category_match,
    add_assertion_disclaimer,
)


class MockDocument:
    """Mock document for testing."""

    def __init__(self, content: str, entity_type: str):
        self.page_content = content
        self.metadata = {"entity_type": entity_type}


@pytest.mark.asyncio
async def test_housing_question_with_housing_sources():
    """Should pass: housing question with housing sources."""
    question = "Где найти общежитие в Вене?"
    answer = "В Вене есть несколько вариантов общежитий. Studentenheim OeAD находится..."
    docs = [
        MockDocument("Общежития в Вене", "housing"),
        MockDocument("Студенческое жилье", "housing"),
    ]

    result = await validate_answer_assertions(answer, docs, question)

    assert result.is_valid
    assert len(result.warnings) == 0


@pytest.mark.asyncio
async def test_housing_question_with_food_sources():
    """Should fail: housing question but only food sources."""
    question = "Где найти общежитие?"
    answer = "Для поиска жилья в Вене рекомендую обратиться в OeAD..."
    docs = [
        MockDocument("Mensa в университете", "food"),
        MockDocument("Студенческие кафетерии", "food"),
    ]

    result = await validate_answer_assertions(answer, docs, question)

    assert not result.is_valid
    assert len(result.warnings) > 0
    assert "housing" in result.warnings[0].lower()


@pytest.mark.asyncio
async def test_food_question_with_mixed_sources():
    """Should pass: food question with general + food sources."""
    question = "Где поесть в университете?"
    answer = "В университете работает Mensa, где можно недорого поесть..."
    docs = [
        MockDocument("Mensa информация", "food"),
        MockDocument("Общая информация", "general"),
    ]

    result = await validate_answer_assertions(answer, docs, question)

    assert result.is_valid
    assert len(result.warnings) == 0


@pytest.mark.asyncio
async def test_generic_answer_no_validation():
    """Should pass: generic answer without specific topic."""
    question = "Расскажи о Вене"
    answer = "Вена - столица Австрии. Здесь расположены университеты..."
    docs = [
        MockDocument("Общая информация", "general"),
    ]

    result = await validate_answer_assertions(answer, docs, question)

    assert result.is_valid


def test_detect_housing_topic():
    """Test housing topic detection."""
    answer = "Для поиска жилья рекомендую общежитие Studentenheim..."
    topic = _detect_answer_topic(answer)
    assert topic == "housing"


def test_detect_food_topic():
    """Test food topic detection."""
    answer = "В университете работает Mensa, где можно недорого питаться..."
    topic = _detect_answer_topic(answer)
    assert topic == "food"


def test_detect_visa_topic():
    """Test visa topic detection."""
    answer = "Для получения визы D необходимо подать документы..."
    topic = _detect_answer_topic(answer)
    assert topic == "visa"


def test_detect_no_topic():
    """Test generic content without specific topic."""
    answer = "Университет находится в центре города."
    topic = _detect_answer_topic(answer)
    assert topic is None


def test_extract_entity_types():
    """Test entity type extraction from documents."""
    docs = [
        MockDocument("content1", "housing"),
        MockDocument("content2", "food"),
        MockDocument("content3", "housing"),
    ]

    entity_types = _extract_entity_types_from_docs(docs)

    assert entity_types == {"housing", "food"}


def test_add_disclaimer():
    """Test disclaimer addition."""
    answer = "Ответ на вопрос"
    warnings = ["Category mismatch detected"]

    result = add_assertion_disclaimer(answer, warnings)

    assert "⚠️" in result
    assert answer in result


def test_no_disclaimer_without_warnings():
    """Test that no disclaimer is added without warnings."""
    answer = "Ответ на вопрос"
    warnings = []

    result = add_assertion_disclaimer(answer, warnings)

    assert result == answer


@pytest.mark.asyncio
async def test_empty_sources():
    """Test behavior with empty sources."""
    question = "Где жить?"
    answer = "Попробуйте найти общежитие..."
    docs = []

    result = await validate_answer_assertions(answer, docs, question)

    # Should pass with a warning about no sources
    assert result.is_valid
    assert "No sources" in result.warnings[0]


@pytest.mark.asyncio
async def test_finance_topic_with_language_sources():
    """Should fail: finance question but language sources."""
    question = "Сколько стоит обучение?"
    answer = "Стоимость обучения составляет 726 евро за семестр..."
    docs = [
        MockDocument("Курсы немецкого языка", "language"),
        MockDocument("Языковые требования", "language"),
    ]

    result = await validate_answer_assertions(answer, docs, question)

    assert not result.is_valid
    assert "finance" in result.warnings[0].lower()
    assert "language" in result.warnings[0].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
