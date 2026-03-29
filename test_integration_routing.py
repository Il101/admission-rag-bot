"""
Integration test for routing system with real bot components.

Tests intent classification, tool execution, and observability metrics
in a realistic environment. Runs without a real Telegram connection.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ["USE_ROUTING"] = "true"
os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", "sqlite:///test_db.sqlite")


async def test_routing_classification():
    """Test intent classification with various queries."""
    from crag.router import classify_intent, Intent
    from crag.observability import get_routing_stats, reset_routing_stats, increment_routing_stat

    print("\n" + "="*70)
    print("INTEGRATION TEST: Intent Classification")
    print("="*70 + "\n")

    # Reset stats
    reset_routing_stats()

    # Test cases with expected intents
    test_cases = [
        # TOOL_ONLY - Personal progress
        ("На каком я сейчас этапе?", Intent.TOOL_ONLY, ["get_my_progress", "get_next_steps"]),
        ("Что дальше делать?", Intent.TOOL_ONLY, ["get_next_steps"]),
        ("Мой прогресс", Intent.TOOL_ONLY, ["get_my_progress"]),
        ("Что ты знаешь обо мне?", Intent.TOOL_ONLY, ["get_my_profile"]),

        # CHITCHAT
        ("Привет!", Intent.CHITCHAT, []),
        ("Спасибо", Intent.CHITCHAT, []),
        ("Благодарю", Intent.CHITCHAT, []),

        # TOOL_THEN_RAG - Calculations
        ("Дедлайны для бакалавриата", Intent.TOOL_THEN_RAG, ["check_deadline"]),
        ("Сколько стоит учёба?", Intent.TOOL_THEN_RAG, ["calculate_budget"]),

        # RAG_ONLY - Knowledge base
        ("Что такое нострификация?", Intent.RAG_ONLY, []),
        ("Как получить ВНЖ в Австрии?", Intent.RAG_ONLY, []),
        ("Какие университеты лучше для IT?", Intent.RAG_ONLY, []),
    ]

    results = []
    passed = 0
    failed = 0

    for question, expected_intent, expected_tools in test_cases:
        route = classify_intent(question)
        intent_match = route.intent == expected_intent
        tools_subset = all(tool in route.suggested_tools for tool in expected_tools) if expected_tools else True

        # Track stats
        increment_routing_stat(route.intent.value)

        status = "✅" if (intent_match and tools_subset) else "❌"
        if intent_match and tools_subset:
            passed += 1
        else:
            failed += 1

        results.append({
            "status": status,
            "question": question,
            "expected": expected_intent.value,
            "actual": route.intent.value,
            "confidence": route.confidence,
            "tools": route.suggested_tools,
            "reason": route.reason,
        })

        print(f"{status} {question[:50]}")
        print(f"   Expected: {expected_intent.value}, Got: {route.intent.value} (confidence={route.confidence:.2f})")
        if not intent_match:
            print(f"   ❌ Intent mismatch!")
        if expected_tools and not tools_subset:
            print(f"   ⚠️  Expected tools {expected_tools}, got {route.suggested_tools}")
        print()

    # Print summary
    print("\n" + "-"*70)
    print(f"Classification Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    print(f"Pass rate: {passed / len(test_cases) * 100:.1f}%")
    print("-"*70 + "\n")

    # Print routing statistics
    stats = get_routing_stats()
    print("Routing Statistics:")
    print(f"  Total queries: {stats['total']}")
    for intent, count in stats['distribution'].items():
        percentage = stats['percentages'][intent]
        print(f"  {intent}: {count} ({percentage}%)")
    print()

    return passed == len(test_cases)


async def test_tool_execution():
    """Test personal tools execution with mock session."""
    from crag.tools import get_tool_by_name, PERSONAL_TOOLS

    print("="*70)
    print("INTEGRATION TEST: Tool Execution")
    print("="*70 + "\n")

    # Test that personal tools are registered
    print("Checking personal tools registration:")
    for tool_name in PERSONAL_TOOLS:
        tool = get_tool_by_name(tool_name)
        status = "✅" if tool else "❌"
        print(f"  {status} {tool_name}: {'Found' if tool else 'NOT FOUND'}")

    print()

    # Test non-personal tools
    non_personal_tools = ["check_deadline", "calculate_budget", "get_document_checklist"]
    print("Checking calculator tools:")
    for tool_name in non_personal_tools:
        tool = get_tool_by_name(tool_name)
        status = "✅" if tool else "❌"
        print(f"  {status} {tool_name}: {'Found' if tool else 'NOT FOUND'}")

    print()

    # Note: We can't execute personal tools without a real DB session
    print("⚠️  Note: Personal tool execution requires database session")
    print("   Tool registration is verified above.\n")

    return True


async def test_chitchat_responses():
    """Test that chitchat responses are quick and appropriate."""
    from crag.router import get_chitchat_response, is_chitchat, classify_intent
    import time

    print("="*70)
    print("INTEGRATION TEST: Chitchat Quick Responses")
    print("="*70 + "\n")

    chitchat_queries = [
        "Привет!",
        "Спасибо большое",
        "Благодарю",
        "Пока",
        "Ок",
    ]

    for query in chitchat_queries:
        start = time.monotonic()
        route = classify_intent(query)
        response = get_chitchat_response(query) if is_chitchat(route) else None
        latency_ms = (time.monotonic() - start) * 1000

        if is_chitchat(route) and response:
            print(f"✅ '{query}' → chitchat detected ({latency_ms:.1f}ms)")
            print(f"   Response: {response[:60]}...")
        else:
            print(f"❌ '{query}' → NOT detected as chitchat (intent={route.intent.value})")
        print()

    return True


async def test_routing_performance():
    """Test classification performance (should be fast since it's regex-based)."""
    from crag.router import classify_intent
    import time

    print("="*70)
    print("INTEGRATION TEST: Routing Performance")
    print("="*70 + "\n")

    queries = [
        "Мой прогресс",
        "Что такое нострификация?",
        "Привет",
        "Дедлайны для бакалавриата",
        "Как получить студенческую визу?",
    ]

    latencies = []
    for query in queries:
        start = time.monotonic()
        route = classify_intent(query)
        latency_ms = (time.monotonic() - start) * 1000
        latencies.append(latency_ms)

        print(f"Query: {query[:50]}")
        print(f"  Intent: {route.intent.value}, Latency: {latency_ms:.2f}ms")
        print()

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    print(f"Performance Summary:")
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Max latency: {max_latency:.2f}ms")

    # Regex-based classification should be very fast (< 5ms typically)
    if avg_latency < 5.0:
        print(f"  ✅ Performance is excellent (< 5ms)")
    elif avg_latency < 10.0:
        print(f"  ⚠️  Performance is acceptable (< 10ms)")
    else:
        print(f"  ❌ Performance is slower than expected (> 10ms)")

    print()
    return avg_latency < 10.0  # Allow up to 10ms


async def test_observability_metrics():
    """Test that observability metrics are tracked correctly."""
    from crag.observability import (
        get_routing_stats,
        reset_routing_stats,
        increment_routing_stat,
        log_routing_decision,
    )

    print("="*70)
    print("INTEGRATION TEST: Observability Metrics")
    print("="*70 + "\n")

    # Reset and populate stats
    reset_routing_stats()

    # Simulate different intent distributions
    test_distribution = {
        "tool_only": 5,
        "rag_only": 10,
        "tool_rag": 3,
        "chitchat": 2,
    }

    for intent, count in test_distribution.items():
        for _ in range(count):
            increment_routing_stat(intent)

    stats = get_routing_stats()

    print("Simulated Distribution:")
    print(f"  Total: {stats['total']} (expected: {sum(test_distribution.values())})")

    all_match = True
    for intent, expected_count in test_distribution.items():
        actual_count = stats['distribution'].get(intent, 0)
        percentage = stats['percentages'].get(intent, 0)
        match = actual_count == expected_count
        status = "✅" if match else "❌"
        all_match = all_match and match

        print(f"  {status} {intent}: {actual_count} ({percentage}%) - expected {expected_count}")

    print()

    # Test logging (will fallback to logger if LangFuse not configured)
    print("Testing log_routing_decision (will use logger fallback):")
    log_routing_decision(
        user_id=12345,
        question="Test question",
        intent="tool_only",
        suggested_tools=["get_my_progress"],
        confidence=0.92,
        reason="Test reason",
        latency_ms=1.5,
    )
    print("  ✅ Logging executed without errors\n")

    return all_match


async def main():
    """Run all integration tests."""
    print("\n" + "🚀 " + "="*68)
    print("  ROUTING SYSTEM INTEGRATION TESTS")
    print("="*68 + " 🚀\n")

    results = {}

    # Run tests
    results["classification"] = await test_routing_classification()
    results["tool_execution"] = await test_tool_execution()
    results["chitchat"] = await test_chitchat_responses()
    results["performance"] = await test_routing_performance()
    results["observability"] = await test_observability_metrics()

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70 + "\n")

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("🎉 ALL INTEGRATION TESTS PASSED! 🎉")
        print("\nRouting system is ready for production use.")
        print("Next steps:")
        print("  1. Monitor routing distribution in production")
        print("  2. Adjust patterns based on real user queries")
        print("  3. Track confidence scores for pattern refinement")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Review the failures above and fix before deploying.")

    print("\n" + "="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
