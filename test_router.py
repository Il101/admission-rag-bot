"""
Test script for intent classification router.

Run with: python test_router.py
"""

from crag.router import classify_intent, Intent


# Test cases with expected intents
TEST_CASES = [
    # Personal Progress (TOOL_ONLY)
    ("Мой прогресс", Intent.TOOL_ONLY, ["get_my_progress"]),
    ("На каком я этапе?", Intent.TOOL_ONLY, ["get_my_progress"]),
    ("Что я уже сделал?", Intent.TOOL_ONLY, ["get_my_progress"]),
    ("Где я сейчас?", Intent.TOOL_ONLY, ["get_my_progress"]),
    ("Мой чек-лист", Intent.TOOL_ONLY, ["get_my_progress"]),
    ("Что дальше делать?", Intent.TOOL_ONLY, ["get_next_steps"]),
    ("Следующие шаги", Intent.TOOL_ONLY, ["get_next_steps"]),
    ("С чего начать?", Intent.TOOL_ONLY, ["get_next_steps"]),
    ("Мой профиль", Intent.TOOL_ONLY, ["get_my_profile"]),
    ("Что ты знаешь обо мне?", Intent.TOOL_ONLY, ["get_my_profile"]),
    ("Мои университеты", Intent.TOOL_ONLY, ["get_my_entities"]),

    # Calculator (TOOL_ONLY or TOOL_THEN_RAG)
    ("Дедлайн подачи в Uni Wien", Intent.TOOL_ONLY, ["check_deadline"]),
    ("Когда подавать в TU Wien?", Intent.TOOL_ONLY, ["check_deadline"]),
    ("Дедлайны для бакалавриата", Intent.TOOL_THEN_RAG, ["check_deadline"]),
    ("Сколько стоит жизнь в Вене?", Intent.TOOL_THEN_RAG, ["calculate_budget"]),
    ("Бюджет на обучение в Граце", Intent.TOOL_THEN_RAG, ["calculate_budget"]),
    ("Сколько дней до дедлайна?", Intent.TOOL_THEN_RAG, ["calculate_days_until"]),

    # Document checklist
    ("Какие документы нужны?", Intent.TOOL_THEN_RAG, ["get_document_checklist"]),
    ("Мне какие документы подготовить?", Intent.TOOL_ONLY, ["get_my_progress", "get_document_checklist"]),
    ("Список документов для поступления", Intent.TOOL_THEN_RAG, ["get_document_checklist"]),

    # Knowledge Base (RAG_ONLY)
    ("Что такое нострификация?", Intent.RAG_ONLY, []),
    ("Как работает Ergänzungsprüfung?", Intent.RAG_ONLY, []),
    ("Расскажи про визу", Intent.RAG_ONLY, []),
    ("Как получить ВНЖ студента?", Intent.RAG_ONLY, []),
    ("Требования для поступления в TU Wien", Intent.RAG_ONLY, []),
    ("Можно ли работать на студенческой визе?", Intent.RAG_ONLY, []),
    ("Чем отличается Uni от FH?", Intent.RAG_ONLY, []),
    ("Процесс подачи документов", Intent.RAG_ONLY, []),
    ("Какие стипендии доступны?", Intent.RAG_ONLY, []),
    ("Где искать жильё в Вене?", Intent.RAG_ONLY, []),

    # Chitchat (CHITCHAT)
    ("Привет", Intent.CHITCHAT, []),
    ("Здравствуй", Intent.CHITCHAT, []),
    ("Спасибо", Intent.CHITCHAT, []),
    ("Благодарю", Intent.CHITCHAT, []),
    ("Пока", Intent.CHITCHAT, []),
    ("До свидания", Intent.CHITCHAT, []),
    ("Окей", Intent.CHITCHAT, []),
    ("Понятно", Intent.CHITCHAT, []),
    ("Хорошо", Intent.CHITCHAT, []),
]


def test_classification():
    """Test intent classification on all test cases."""
    print("=" * 80)
    print("TESTING INTENT CLASSIFICATION")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for question, expected_intent, expected_tools in TEST_CASES:
        result = classify_intent(question)

        # Check intent match
        intent_match = result.intent == expected_intent

        # Check if at least one expected tool is suggested (for tool-based intents)
        tools_match = True
        if expected_tools:
            tools_match = any(tool in result.suggested_tools for tool in expected_tools)

        success = intent_match and tools_match

        if success:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"

        print(f"{status} | '{question}'")
        print(f"  Expected: {expected_intent.value} {expected_tools}")
        print(f"  Got:      {result.intent.value} {result.suggested_tools}")
        print(f"  Reason:   {result.reason}")
        print(f"  Confidence: {result.confidence:.2f}")

        if not success:
            if not intent_match:
                print(f"  ⚠️  Intent mismatch!")
            if expected_tools and not tools_match:
                print(f"  ⚠️  Tools mismatch!")
        print()

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print(f"Pass rate: {100 * passed / len(TEST_CASES):.1f}%")
    print("=" * 80)

    return passed, failed


def test_edge_cases():
    """Test edge cases and complex queries."""
    print("\n" + "=" * 80)
    print("TESTING EDGE CASES")
    print("=" * 80)
    print()

    edge_cases = [
        "Какие дедлайны подачи для Uni Wien на бакалавриат?",
        "Сколько стоит обучение и жизнь в Австрии?",
        "Мне нужен чек-лист документов, что уже сделал и что дальше?",
        "Как получить визу и сколько это стоит?",
        "Какие университеты мне подходят и когда туда подавать?",
    ]

    for question in edge_cases:
        result = classify_intent(question)
        print(f"Question: '{question}'")
        print(f"  Intent: {result.intent.value}")
        print(f"  Tools: {result.suggested_tools}")
        print(f"  Reason: {result.reason}")
        print(f"  Confidence: {result.confidence:.2f}")
        print()


def test_false_positives():
    """Test potential false positives (questions that shouldn't match personal progress)."""
    print("=" * 80)
    print("TESTING FALSE POSITIVES")
    print("=" * 80)
    print()

    false_positive_tests = [
        ("Мой вопрос про документы", Intent.RAG_ONLY),  # "мой" but not about progress
        ("На каком уровне нужен немецкий?", Intent.RAG_ONLY),  # "на каком" but not about user stage
        ("Что такое мой Aufenthalt?", Intent.RAG_ONLY),  # "мой" but asking definition
    ]

    for question, expected_intent in false_positive_tests:
        result = classify_intent(question)
        match = result.intent == expected_intent
        status = "✅" if match else "❌"

        print(f"{status} '{question}'")
        print(f"  Expected: {expected_intent.value}")
        print(f"  Got: {result.intent.value} (confidence={result.confidence:.2f})")
        print()


if __name__ == "__main__":
    passed, failed = test_classification()
    test_edge_cases()
    test_false_positives()

    print("\n" + "=" * 80)
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {failed} tests failed. Review patterns in crag/router.py")
    print("=" * 80)
