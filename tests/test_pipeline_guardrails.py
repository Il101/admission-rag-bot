from crag.pipeline import apply_entity_grounding_guardrails
from crag.simple_rag import Document


def _doc(text: str, topic: str = "") -> Document:
    metadata = {"source": "test"}
    if topic:
        metadata["topic"] = topic
    return Document(page_content=text, metadata=metadata)


def test_housing_question_drops_food_only_chunks():
    question = "Есть ли в общежитии вегетарианское меню?"
    docs = [
        _doc("Studentenheim: стоимость и условия заселения.", topic="housing"),
        _doc("Mensa (столовая вуза): обед от €5-7.", topic="student-budget"),
        _doc("WG и Studentenheim: сравнение по аренде.", topic="housing"),
    ]

    filtered = apply_entity_grounding_guardrails(question, docs)

    assert len(filtered) == 2
    assert all("mensa" not in d.page_content.lower() for d in filtered)
    assert any("studentenheim" in d.page_content.lower() for d in filtered)


def test_food_intent_keeps_food_chunks():
    question = "Где есть вегетарианское меню и сколько стоит Mensa?"
    docs = [
        _doc("Studentenheim: стоимость и условия.", topic="housing"),
        _doc("Mensa (столовая вуза): есть вегетарианские опции.", topic="student-budget"),
    ]

    filtered = apply_entity_grounding_guardrails(question, docs)

    assert len(filtered) == 2


def test_guardrail_fail_open_when_everything_filtered():
    question = "Как снять общежитие в Вене?"
    docs = [
        _doc("Mensa (столовая вуза): ужин около €12.", topic="student-budget"),
        _doc("Кафе и рестораны в кампусе.", topic="student-budget"),
    ]

    filtered = apply_entity_grounding_guardrails(question, docs)

    # Fail-open behavior: if we removed all docs, keep originals to avoid empty context.
    assert len(filtered) == len(docs)
