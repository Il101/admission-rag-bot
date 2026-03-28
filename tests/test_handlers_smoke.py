from bot.handlers import rag as rag_handlers
from bot.utils import remove_bot_command


def test_parse_suggested_buttons_valid_json():
    text = '{"answer":"Привет","suggested_questions":["Q1","Q2"]}'
    answer, suggested = rag_handlers.parse_suggested_buttons(text)
    assert answer == "Привет"
    assert suggested == ["Q1", "Q2"]


def test_parse_suggested_buttons_invalid_answer_fallback():
    text = '{"answer":"...","suggested_questions":["Q1"]}'
    answer, suggested = rag_handlers.parse_suggested_buttons(text)
    assert "не удалось сформировать ответ" in answer.lower()
    assert suggested == ["Q1"]


def test_build_user_filters_from_onboarding():
    filters = rag_handlers._build_user_filters(
        {"country": "россия", "targetLevel": "master"}
    )
    assert filters == {"country_scope": "RU", "level": "master"}


def test_build_user_filters_unknown_country():
    filters = rag_handlers._build_user_filters({"country": "марс"})
    assert filters == {}


def test_remove_bot_command_smoke():
    assert remove_bot_command("/ans@testbot Как дела?", "ans", "@testbot") == "Как дела?"


def test_deadline_table_renders_as_cards():
    text = (
        "📋 Бакалавриат\n\n"
        "Учебное заведение\n"
        "Winter 2026/27\n"
        "Summer 2027\n"
        "Universität Innsbruck (UIBK)\n"
        "1 апреля — 15 мая 2026\n"
        "1 сентября — 15 октября 2026\n"
    )
    out = rag_handlers._format_deadline_tables_as_cards(text)
    assert "📍 Universität Innsbruck (UIBK)" in out
    assert "• Winter 2026/27: 1 апреля — 15 мая 2026" in out
    assert "• Summer 2027: 1 сентября — 15 октября 2026" in out


def test_confirmed_fact_guardrail_rewrites_unconfirmed_claim():
    text = "✅ Что уже готово? Вы уже получили Zulassungsbescheid и можно двигаться дальше."
    memory = {"journey_state": {"_facts": {"zulassungsbescheid": "not_done"}}}
    out = rag_handlers._apply_confirmed_fact_guardrails(text, memory)
    assert "Вы уже получили Zulassungsbescheid" not in out
    assert "получение Zulassungsbescheid — обязательный шаг" in out


def test_confirmed_fact_guardrail_keeps_confirmed_claim():
    text = "Вы уже получили Zulassungsbescheid."
    memory = {"journey_state": {"_facts": {"zulassungsbescheid": "done"}}}
    out = rag_handlers._apply_confirmed_fact_guardrails(text, memory)
    assert out == text
