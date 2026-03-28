"""Tests for tag-set layer."""

import pytest
from crag.tag_set_layer import (
    auto_tag_chunk,
    extract_query_tags,
    calculate_tag_boost,
    tag_based_reranking,
    add_tags_to_metadata,
    DomainTag,
)


def test_auto_tag_dormitory():
    """Should tag dormitory content correctly."""
    content = "Studentenheim OeAD предлагает общежития для студентов"
    entity_type = "housing"

    tags = auto_tag_chunk(content, entity_type)

    assert "housing:dormitory" in tags
    assert "entity:housing" in tags


def test_auto_tag_mensa():
    """Should tag mensa/food content correctly."""
    content = "На кампусе работает Mensa, где студенты могут питаться"
    entity_type = "food"

    tags = auto_tag_chunk(content, entity_type)

    assert "food:mensa" in tags
    assert "entity:food" in tags


def test_auto_tag_tuition():
    """Should tag tuition costs correctly."""
    content = "Плата за обучение составляет 726 евро за семестр (Studienbeitrag)"
    entity_type = "finance"

    tags = auto_tag_chunk(content, entity_type)

    assert "finance:tuition" in tags
    assert "entity:finance" in tags


def test_auto_tag_visa():
    """Should tag visa requirements correctly."""
    content = "Для получения Aufenthaltstitel (вида на жительство) подайте документы"
    entity_type = "visa"

    tags = auto_tag_chunk(content, entity_type)

    # May have multiple visa tags
    assert any("visa:" in tag for tag in tags)
    assert "entity:visa" in tags


def test_auto_tag_language():
    """Should tag language-related content."""
    content = "Требуется сертификат немецкого языка (German language requirement)"
    entity_type = "language"

    tags = auto_tag_chunk(content, entity_type)

    assert any("language:" in tag for tag in tags)
    assert "entity:language" in tags


def test_extract_query_tags_housing():
    """Should extract housing tags from query."""
    query = "Где найти общежитие в Вене?"

    tags = extract_query_tags(query)

    assert "housing:dormitory" in tags


def test_extract_query_tags_food():
    """Should extract food tags from query."""
    query = "Где я могу питаться? Есть ли Mensa?"

    tags = extract_query_tags(query)

    assert "food:mensa" in tags


def test_extract_query_tags_multiple():
    """Should extract multiple tags from complex query."""
    query = "Сколько стоит обучение и могу ли я получить стипендию?"

    tags = extract_query_tags(query)

    # Query mentions both tuition and scholarships
    assert "finance:scholarships" in tags


def test_extract_query_tags_none():
    """Should return empty list for generic query."""
    query = "Расскажи мне о Вене"

    tags = extract_query_tags(query)

    # Generic query might not have specific tags
    assert len(tags) == 0 or all("general:" in tag for tag in tags)


def test_calculate_tag_boost_full_match():
    """Should give max boost for full tag match."""
    chunk_tags = ["housing:dormitory", "entity:housing"]
    query_tags = ["housing:dormitory"]

    boost = calculate_tag_boost(chunk_tags, query_tags)

    assert boost > 1.0
    assert boost <= 1.5


def test_calculate_tag_boost_partial_match():
    """Should give partial boost for partial match."""
    chunk_tags = ["housing:dormitory", "entity:housing"]
    query_tags = ["housing:dormitory", "housing:costs"]

    boost = calculate_tag_boost(chunk_tags, query_tags)

    assert 1.0 < boost <= 1.5


def test_calculate_tag_boost_no_match():
    """Should not boost for no match."""
    chunk_tags = ["housing:dormitory"]
    query_tags = ["food:mensa"]

    boost = calculate_tag_boost(chunk_tags, query_tags)

    assert boost == 1.0


def test_calculate_tag_boost_empty_tags():
    """Should return 1.0 for empty tags."""
    boost1 = calculate_tag_boost([], ["food:mensa"])
    boost2 = calculate_tag_boost(["housing:dormitory"], [])

    assert boost1 == 1.0
    assert boost2 == 1.0


def test_tag_based_reranking_simple():
    """Should rerank documents based on tag match."""
    docs = [
        {
            "content": "Mensa info",
            "score": 0.5,
            "metadata": {"tags": ["food:mensa"]},
        },
        {
            "content": "Housing info",
            "score": 0.4,  # Lower initial score
            "metadata": {"tags": ["housing:dormitory"]},
        },
    ]
    query = "Где питаться? Есть ли Mensa?"

    ranked = tag_based_reranking(docs, query, factor=0.5)

    # Mensa doc should rank first after tag boost
    assert ranked[0][0]["metadata"]["tags"][0] == "food:mensa"


def test_tag_based_reranking_preserves_docs():
    """Reranking should preserve document objects."""
    docs = [
        {
            "content": "Doc 1",
            "score": 0.5,
            "metadata": {"tags": ["food:mensa"]},
        },
    ]
    query = "Mensa"

    ranked = tag_based_reranking(docs, query)

    assert len(ranked) == 1
    assert ranked[0][0]["content"] == "Doc 1"


def test_add_tags_to_metadata_housing():
    """Should add tags to housing metadata."""
    metadata = {"source": "housing.md", "entity_type": "housing"}
    content = "Studentenheim для поиска общежития"

    updated = add_tags_to_metadata(metadata, content)

    assert "tags" in updated
    assert "housing:dormitory" in updated["tags"]
    assert "entity:housing" in updated["tags"]


def test_add_tags_to_metadata_preserves_existing():
    """Should preserve existing metadata fields."""
    metadata = {
        "source": "test.md",
        "entity_type": "food",
        "university": "TU Wien",
        "custom_field": "value",
    }
    content = "Mensa на кампусе"

    updated = add_tags_to_metadata(metadata, content)

    assert updated["source"] == "test.md"
    assert updated["entity_type"] == "food"
    assert updated["university"] == "TU Wien"
    assert updated["custom_field"] == "value"
    assert "tags" in updated


def test_tag_boost_factor_zero():
    """With factor=0, tags should not affect scoring."""
    docs = [
        {
            "content": "Mensa",
            "score": 0.5,
            "metadata": {"tags": ["food:mensa"]},
        },
    ]
    query = "Mensa"

    ranked = tag_based_reranking(docs, query, factor=0.0)

    # Score should remain unchanged when factor=0
    assert ranked[0][1] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
