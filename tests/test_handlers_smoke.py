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

