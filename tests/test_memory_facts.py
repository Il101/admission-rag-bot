from bot import memory as mem


def test_extract_fact_updates_detects_not_done():
    q = "Я Zulassungsbescheid еще не получал, что делать?"
    updates = mem._extract_fact_updates(q)
    assert updates.get("zulassungsbescheid") == "not_done"


def test_extract_fact_updates_detects_done():
    q = "Я уже получил Zulassungsbescheid и готов подавать на визу."
    updates = mem._extract_fact_updates(q)
    assert updates.get("zulassungsbescheid") == "done"


def test_build_memory_context_shows_confirmed_facts():
    journey_state = {
        "documents_prep": "discussed",
        "application": "pending",
        "_facts": {"zulassungsbescheid": "not_done"},
    }
    ctx = mem.build_memory_context(journey_state, "test summary", {"countryScope": "RU"})
    assert "Подтвержденные факты пользователя" in ctx
    assert "еще не получен" in ctx
