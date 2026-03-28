"""Tag-Set Layer for Domain-Specific Boosting.

Implements a closed vocabulary of domain tags used to:
1. Categorize chunks with multiple tags
2. Boost retrieval of tagged chunks with matching query tags
3. Prevent irrelevant results through tag filtering

Tags are hierarchical and domain-specific for student information.
"""

import logging
from typing import List, Set, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DomainTag(str, Enum):
    """Closed vocabulary of domain tags."""

    # Housing tags
    HOUSING_DORMITORY = "housing:dormitory"
    HOUSING_PRIVATE = "housing:private"
    HOUSING_REGISTRATION = "housing:registration"
    HOUSING_COSTS = "housing:costs"

    # Food tags
    FOOD_MENSA = "food:mensa"
    FOOD_CAFETERIA = "food:cafeteria"
    FOOD_DIETARY = "food:dietary"
    FOOD_COSTS = "food:costs"

    # Finance tags
    FINANCE_TUITION = "finance:tuition"
    FINANCE_FEES = "finance:fees"
    FINANCE_SCHOLARSHIPS = "finance:scholarships"
    FINANCE_HOUSING_COSTS = "finance:housing_costs"

    # Visa/Legal tags
    VISA_APPLICATION = "visa:application"
    VISA_REQUIREMENTS = "visa:requirements"
    VISA_RESIDENCE = "visa:residence"
    VISA_EXTENSION = "visa:extension"

    # Language tags
    LANGUAGE_GERMAN = "language:german"
    LANGUAGE_ENGLISH = "language:english"
    LANGUAGE_REQUIREMENTS = "language:requirements"
    LANGUAGE_COURSES = "language:courses"

    # Admission tags
    ADMISSION_APPLICATION = "admission:application"
    ADMISSION_REQUIREMENTS = "admission:requirements"
    ADMISSION_DEADLINES = "admission:deadlines"
    ADMISSION_DOCUMENTS = "admission:documents"

    # General tags
    GENERAL_INFO = "general:info"
    GENERAL_FAQ = "general:faq"
    GENERAL_CONTACTS = "general:contacts"


# Keyword patterns for auto-tagging
TAG_KEYWORDS = {
    DomainTag.HOUSING_DORMITORY: ["общежит", "studentenheim", "wg", "дом студента", "студгородок"],
    DomainTag.HOUSING_PRIVATE: ["квартир", "аренд", "wohnung", "miet", "приватн"],
    DomainTag.HOUSING_REGISTRATION: ["мелдецеттель", "прописк", "регистр", "зареги"],
    DomainTag.HOUSING_COSTS: ["стоимость жилья", "цена квартир", "мієт"],
    DomainTag.FOOD_MENSA: ["mensa", "меnса", "столов"],
    DomainTag.FOOD_CAFETERIA: ["кафе", "буфет", "кафетери", "cafeteria"],
    DomainTag.FOOD_DIETARY: ["веган", "вегетари", "allergic", "диетич"],
    DomainTag.FOOD_COSTS: ["стоимость еды", "цена обед", "meal plan"],
    DomainTag.FINANCE_TUITION: ["плата за обучение", "tuition", "studienbeitrag"],
    DomainTag.FINANCE_FEES: ["сбор", "пошлин", "gebühr", "fee"],
    DomainTag.FINANCE_SCHOLARSHIPS: ["стипенди", "scholarship", "grant", "финансовая помощь"],
    DomainTag.FINANCE_HOUSING_COSTS: ["жилищные расходы", "housing costs", "mietbeihilfe"],
    DomainTag.VISA_APPLICATION: ["заявк", "application", "antrag", "поступл"],
    DomainTag.VISA_REQUIREMENTS: ["требовани", "requirement", "erfordernis"],
    DomainTag.VISA_RESIDENCE: ["вид на жительство", "residence permit", "aufenthaltstitel"],
    DomainTag.VISA_EXTENSION: ["продлени", "extension", "verlängerung"],
    DomainTag.LANGUAGE_GERMAN: ["немецк", "german", "deutsch"],
    DomainTag.LANGUAGE_ENGLISH: ["английск", "english"],
    DomainTag.LANGUAGE_REQUIREMENTS: ["язык требовани", "language requirement", "sprachanfordung"],
    DomainTag.LANGUAGE_COURSES: ["курс языка", "language course", "sprachkurs"],
    DomainTag.ADMISSION_APPLICATION: ["поступ в", "admission", "bewerbung", "заявк"],
    DomainTag.ADMISSION_REQUIREMENTS: ["требовани", "prerequisite", "voraussetzung"],
    DomainTag.ADMISSION_DEADLINES: ["дедлайн", "deadline", "termine"],
    DomainTag.ADMISSION_DOCUMENTS: ["документ", "document", "unterlagen"],
    DomainTag.GENERAL_FAQ: ["часто задаваемые", "faq", "вопрос ответ"],
    DomainTag.GENERAL_CONTACTS: ["контакт", "адрес", "телефон", "contact", "phone"],
}


def auto_tag_chunk(content: str, entity_type: Optional[str] = None) -> List[str]:
    """Automatically assign tags to a chunk based on content and entity type.

    Args:
        content: Chunk text content
        entity_type: The entity type category (housing, food, visa, etc.)

    Returns:
        List of applicable tags
    """
    tags = []
    content_lower = content.lower()

    # Scan all keyword patterns
    for tag, keywords in TAG_KEYWORDS.items():
        # All keywords in a tag must be present for a match
        # This avoids false positives
        if any(kw in content_lower for kw in keywords):
            tags.append(tag.value)

    # Always add entity type as a base tag
    if entity_type:
        tags.append(f"entity:{entity_type}")

    return list(set(tags))  # Remove duplicates


def extract_query_tags(query: str) -> List[str]:
    """Extract tags from user query text.

    Args:
        query: User question text

    Returns:
        List of inferred tags
    """
    tags = []
    query_lower = query.lower()

    # Check for keyword patterns
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            tags.append(tag.value)

    return list(set(tags))


def calculate_tag_boost(chunk_tags: List[str], query_tags: List[str]) -> float:
    """Calculate boost factor based on tag overlap.

    Args:
        chunk_tags: Tags assigned to the chunk
        query_tags: Tags inferred from query

    Returns:
        Boost factor (1.0 = no boost, >1.0 = should rank higher)
    """
    if not chunk_tags or not query_tags:
        return 1.0

    # Convert to sets for intersection
    chunk_tag_set = set(chunk_tags)
    query_tag_set = set(query_tags)

    # Find matching tags
    matches = chunk_tag_set & query_tag_set

    if not matches:
        return 1.0

    # Boost is 1 + (percentage of query tags that matched)
    # Max boost of 1.5 for full match
    boost = 1.0 + min(len(matches) / len(query_tag_set), 0.5)

    return boost


def tag_based_reranking(
    docs: List[dict],
    query: str,
    factor: float = 0.2,
) -> List[tuple]:
    """Rerank documents based on tag matching.

    Args:
        docs: List of documents with 'content', 'metadata', 'score' keys
        query: Original query text
        factor: Weight factor for tag boost (0.0-1.0)
              0.0 = ignore tags, 1.0 = heavily weight tags

    Returns:
        List of (doc, adjusted_score) tuples, sorted by adjusted score
    """
    query_tags = extract_query_tags(query)

    ranked = []
    for doc in docs:
        original_score = doc.get("score", 0.5)
        chunk_tags = doc.get("metadata", {}).get("tags", [])

        # Calculate tag boost
        boost = calculate_tag_boost(chunk_tags, query_tags)

        # Apply boost to score
        # adjusted_score = original_score * boost_multiplier
        # where boost_multiplier = 1 + (boost - 1.0) * factor
        # factor controls influence: 0.0 = ignore tags, 1.0 = full boost
        boost_multiplier = 1 + (boost - 1.0) * factor
        adjusted_score = original_score * boost_multiplier

        ranked.append((doc, adjusted_score))

    # Sort by adjusted score, descending
    ranked.sort(key=lambda x: x[1], reverse=True)

    return ranked


def add_tags_to_metadata(metadata: dict, content: str) -> dict:
    """Add tags to chunk metadata.

    Args:
        metadata: Existing metadata dict
        content: Chunk content

    Returns:
        Updated metadata with 'tags' field
    """
    entity_type = metadata.get("entity_type")
    tags = auto_tag_chunk(content, entity_type)

    metadata["tags"] = tags
    return metadata
