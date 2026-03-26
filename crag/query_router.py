"""Query Router for YAML Facts vs Markdown Narratives.

This module classifies user queries to determine whether they need:
- FACT retrieval (structured YAML data: deadlines, fees, requirements, quotas)
- NARRATIVE retrieval (markdown guides: processes, advice, step-by-step)
- HYBRID retrieval (both types needed)

The router uses keyword patterns and optional LLM classification for ambiguous cases.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query classification types."""
    FACT = "fact"           # Specific data: deadlines, fees, requirements
    NARRATIVE = "narrative" # Processes, guides, explanations
    HYBRID = "hybrid"       # Needs both fact and narrative context


@dataclass
class RoutingResult:
    """Result of query classification."""
    query_type: QueryType
    confidence: float  # 0.0 - 1.0
    detected_entities: dict  # university, topic, etc.
    reason: str


# ── Keyword patterns for FACT queries ────────────────────────────────────────

FACT_PATTERNS = [
    # Deadlines
    (r'\b(дедлайн|срок|когда\s+(подат|сдат|регистрац))', 'deadline'),
    (r'\b(deadline|due\s*date|submission\s*date)', 'deadline'),
    (r'\bдо\s+какого\s+(числа|срока)', 'deadline'),
    (r'\b(последний\s+день|крайний\s+срок)', 'deadline'),

    # Fees and costs
    (r'\b(стоимость|цена|плата|сколько\s+стоит)', 'cost'),
    (r'\b(tuition|fee|cost|price)', 'cost'),
    (r'\b(€|евро|euro)', 'cost'),
    (r'\b(семестр|semester)\b.*\b(плат|стоит)', 'cost'),
    (r'\b(ÖH-Beitrag|Studienbeitrag)', 'cost'),

    # Language requirements
    (r'\b(уровень\s+(немецк|язык)|B[12]|C[12]|A[12])\b', 'language'),
    (r'\b(language\s+level|german\s+level|minimum\s+level|required\s+level)\b', 'language'),
    (r'\b(сертификат|TestDaF|ÖSD|Goethe|telc|DSH|TDN[345])', 'language'),
    (r'\b(какой\s+уровень|минимальн\w+\s+уровень)', 'language'),

    # Quotas
    (r'\b(квота|места|мест\b|places|quota)', 'quota'),
    # non-EU is only a quota signal when mentioned with quota/places/% context
    (
        r'(\b(non-eu|не-eu)\b.*\b(квот\w*|quota|мест\w*|процент|%)\b)|'
        r'(\b(квот\w*|quota|мест\w*|процент|%)\b.*\b(non-eu|не-eu)\b)',
        'quota'
    ),
    (r'\b(процент|percentage|%)', 'quota'),

    # Specific requirements
    (r'\b(требовани[яе]|requirement)', 'requirement'),
    (r'\b(документ\w*\s+(нуж|треб|нужн))', 'requirement'),
    (r'\b(какие\s+документ\w*|список\s+документ\w*|перечень\s+документ\w*)', 'requirement'),
    (r'\b(апостиль|apostille|легализация)', 'requirement'),

    # Contact info
    (r'\b(адрес|email|телефон|contact|phone)', 'contact'),

    # Exam dates
    (r'\b(MedAT|вступительн\w+\s+экзамен|entrance\s+exam)', 'exam'),
    (r'\b(дата\s+(теста|экзамена)|exam\s+date)', 'exam'),

    # Numbers / specific data
    (r'\b(сколько|какое\s+количество|how\s+many|how\s+much)', 'quantity'),
]

# ── Keyword patterns for NARRATIVE queries ───────────────────────────────────

NARRATIVE_PATTERNS = [
    # Process / how-to
    (r'\b(как\s+(подат\w*|поступ\w*|оформ\w*|получ\w*|сдел\w*))', 'process'),
    (r'\b(как\s+(подготов\w*|сэконом\w*|повыс\w*|улучш\w*|дойти|усп\w*|напис\w*|выбр\w*))', 'process'),
    (r'\bкак\b[^?.!,;:\n]{0,40}\b(подготов\w*|сэконом\w*|повыс\w*|улучш\w*|дойти|усп\w*|напис\w*|выбр\w*|получ\w*)', 'process'),
    (r'\b(что\s+делать)', 'process'),
    (r'\b(how\s+to|process|procedure|steps)', 'process'),
    (r'\b(how\s+should\s+i|how\s+can\s+i|best\s+strategy)', 'process'),
    (r'\b(пошагов|шаг\s+за\s+шагом|step\s*by\s*step)', 'process'),

    # Explanations
    (r'\b(что\s+такое|что\s+значит|explain|what\s+is|what\s+does|meaning\s+of)', 'explanation'),
    (r'\b(расскажи|объясни|подробн)', 'explanation'),

    # Comparison / advice
    (r'\b(лучше|хуже|сравн|выбра|recommend|which\s+is\s+better)', 'advice'),
    (r'\b(стоит\s+ли|should\s+I)', 'advice'),
    (r'\b(плюсы|минусы|pros|cons)', 'advice'),

    # Strategy
    (r'\b(стратеги|план\s+действий|checklist|чеклист)', 'strategy'),
    (r'\b(подготов\w+\s+к)', 'strategy'),

    # VWU / preparation path
    (r'\b(VWU|vorstudienlehrgang|подготовительн)', 'vwu_process'),

    # Visa / residence
    (r'\b(виза|visa|ВНЖ|residence\s+permit|aufenthalt)', 'visa_process'),
]

# ── University name patterns ─────────────────────────────────────────────────

UNIVERSITY_PATTERNS = {
    'tu-wien': [r'\bTU\s*Wien\b', r'\bТУ\s*Вена?\b', r'\bтехнически[йи]\s+университет\s+вен'],
    'uni-wien': [r'\bUni\s*Wien\b', r'\bУни\s*Вена?\b', r'\bвенски[йи]\s+университет\b', r'\bунивер\w+\s+вены\b'],
    'wu-wien': [r'\bWU\s*Wien\b', r'\bwirtschaftsuniversität\b', r'\bэкономическ\w+\s+универ'],
    'meduni-wien': [r'\bMedUni\s*Wien\b', r'\bмедицинск\w+\s+университет\s+вен'],
    'tu-graz': [r'\bTU\s*Graz\b', r'\bТУ\s*Грац'],
    'uni-graz': [r'\bUni\s*Graz\b', r'\bУни\s*Грац', r'\bграц\w+\s+университет'],
    'uni-innsbruck': [r'\bUni\s*Innsbruck\b', r'\bинсбрук\w+\s+университет'],
    'jku-linz': [r'\bJKU\b', r'\bLinz\b.*\buniversit', r'\bлинц'],
    'uni-salzburg': [r'\bUni\s*Salzburg\b', r'\bPLUS\b', r'\bзальцбург'],
}


def _match_patterns(text: str, patterns: list) -> list[tuple[str, str]]:
    """Match text against a list of (regex, category) patterns.

    Returns list of (matched_text, category) tuples.
    """
    text_lower = text.lower()
    matches = []
    for pattern, category in patterns:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            matches.append((match.group(0), category))
    return matches


def _detect_universities(text: str) -> list[str]:
    """Detect university references in the query."""
    text_lower = text.lower()
    detected = []
    for uni_id, patterns in UNIVERSITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append(uni_id)
                break
    return detected


def classify_query(query: str) -> RoutingResult:
    """Classify a query into FACT, NARRATIVE, or HYBRID type.

    Uses keyword pattern matching for fast, deterministic classification.

    Args:
        query: User's question/query text

    Returns:
        RoutingResult with query type, confidence, and detected entities
    """
    if not query or not query.strip():
        return RoutingResult(
            query_type=QueryType.NARRATIVE,
            confidence=0.5,
            detected_entities={},
            reason="Empty query"
        )

    # Match patterns
    fact_matches = _match_patterns(query, FACT_PATTERNS)
    narrative_matches = _match_patterns(query, NARRATIVE_PATTERNS)
    universities = _detect_universities(query)

    # Calculate scores
    fact_score = len(fact_matches)
    narrative_score = len(narrative_matches)

    # Boost fact score if specific university mentioned with data request.
    # Do not boost when narrative intent is also present to avoid FACT overbias on hybrid queries.
    if universities and fact_matches and not narrative_matches:
        fact_score += 1.5

    # Extract categories for reason
    fact_categories = list(set(cat for _, cat in fact_matches))
    narrative_categories = list(set(cat for _, cat in narrative_matches))
    weak_narrative_only = (
        narrative_categories
        and set(narrative_categories).issubset({'explanation'})
    )

    # Build entities dict
    entities = {}
    if universities:
        entities['universities'] = universities
    if fact_categories:
        entities['fact_topics'] = fact_categories
    if narrative_categories:
        entities['narrative_topics'] = narrative_categories

    mixed_intent_connectors = bool(
        re.search(r'\b(и|and)\b.{0,25}\b(как|how)\b', query.lower())
    )

    # Decision logic
    if fact_score > 0 and narrative_score > 0:
        # "расскажи/объясни про X" with a concrete fact topic is usually factual.
        if weak_narrative_only:
            query_type = QueryType.FACT
            confidence = min(0.9, 0.65 + fact_score * 0.1)
            reason = f"Fact query with weak narrative signal ({fact_categories})"
        # Hybrid: both fact data and explanation needed
        elif fact_score > narrative_score * 2.0:
            query_type = QueryType.FACT
            confidence = min(0.9, 0.6 + fact_score * 0.1)
            reason = f"Strong fact signals ({fact_categories}) override narrative ({narrative_categories})"
        elif narrative_score > fact_score * 2.0:
            query_type = QueryType.NARRATIVE
            confidence = min(0.9, 0.6 + narrative_score * 0.1)
            reason = f"Strong narrative signals ({narrative_categories}) override fact ({fact_categories})"
        else:
            query_type = QueryType.HYBRID
            confidence = 0.7
            reason = f"Mixed signals: fact ({fact_categories}), narrative ({narrative_categories})"
    elif fact_score > 0:
        # Questions like "какой уровень ... и как его получить" are hybrid even if
        # narrative cue was not captured by patterns.
        if mixed_intent_connectors:
            query_type = QueryType.HYBRID
            confidence = 0.65
            reason = f"Fact + mixed-intent connector detected ({fact_categories})"
        else:
            query_type = QueryType.FACT
            confidence = min(0.95, 0.7 + fact_score * 0.1)
            reason = f"Fact query: {fact_categories}"
    elif narrative_score > 0:
        query_type = QueryType.NARRATIVE
        confidence = min(0.95, 0.7 + narrative_score * 0.1)
        reason = f"Narrative query: {narrative_categories}"
    else:
        # Default to hybrid for ambiguous queries
        query_type = QueryType.HYBRID
        confidence = 0.5
        reason = "No clear signals, defaulting to hybrid"

    logger.debug(
        f"Query classified: '{query[:50]}...' -> {query_type.value} "
        f"(confidence={confidence:.2f}, reason={reason})"
    )

    return RoutingResult(
        query_type=query_type,
        confidence=confidence,
        detected_entities=entities,
        reason=reason
    )


async def classify_query_with_llm(
    query: str,
    llm_provider,
    fallback_result: Optional[RoutingResult] = None
) -> RoutingResult:
    """Use LLM to classify ambiguous queries.

    Called when keyword-based classification has low confidence.
    Falls back to keyword result if LLM fails.

    Args:
        query: User's question
        llm_provider: LLM provider instance
        fallback_result: Result from keyword classification to use on failure

    Returns:
        RoutingResult with LLM-based classification
    """
    if fallback_result is None:
        fallback_result = classify_query(query)

    prompt = f"""Классифицируй вопрос пользователя о поступлении в австрийские вузы.

ВОПРОС: {query}

ТИПЫ:
- FACT: Вопрос требует конкретных данных (дедлайны, стоимость, требования к языку, квоты, адреса, даты экзаменов)
- NARRATIVE: Вопрос требует объяснения процесса, совета, пошагового руководства
- HYBRID: Нужны и конкретные данные, и объяснение процесса

Примеры:
- "Какой дедлайн подачи в TU Wien?" → FACT
- "Как подать документы в Uni Wien?" → NARRATIVE
- "Какой уровень немецкого нужен для TU Wien и как его получить?" → HYBRID

Ответь ОДНИМ словом: FACT, NARRATIVE или HYBRID"""

    try:
        response = await llm_provider.generate(prompt=prompt, system_prompt=None)
        response = (response or "").strip().upper()

        if "FACT" in response:
            query_type = QueryType.FACT
        elif "NARRATIVE" in response:
            query_type = QueryType.NARRATIVE
        else:
            query_type = QueryType.HYBRID

        return RoutingResult(
            query_type=query_type,
            confidence=0.85,
            detected_entities=fallback_result.detected_entities,
            reason=f"LLM classification: {response}"
        )
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}, using keyword fallback")
        return fallback_result


def get_search_weights(routing_result: RoutingResult) -> dict:
    """Get retrieval weights based on query classification.

    Returns weights for YAML facts vs Markdown narratives search.

    Args:
        routing_result: Result from query classification

    Returns:
        Dict with 'facts_weight' and 'narratives_weight' (sum to 1.0)
    """
    if routing_result.query_type == QueryType.FACT:
        return {'facts_weight': 0.8, 'narratives_weight': 0.2}
    elif routing_result.query_type == QueryType.NARRATIVE:
        return {'facts_weight': 0.2, 'narratives_weight': 0.8}
    else:  # HYBRID
        return {'facts_weight': 0.5, 'narratives_weight': 0.5}
