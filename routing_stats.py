"""
Routing statistics monitor - displays live stats from observability module.

Usage:
    python3 routing_stats.py        # Show current stats
    python3 routing_stats.py --reset  # Reset stats
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crag.observability import get_routing_stats, reset_routing_stats


def format_bar(value: float, max_value: float, width: int = 30) -> str:
    """Create a text-based bar chart."""
    if max_value == 0:
        return "│" + " " * width + "│"

    filled = int((value / max_value) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"│{bar}│"


def display_stats():
    """Display routing statistics in a nice format."""
    stats = get_routing_stats()

    if stats["total"] == 0:
        print("\n📊 Routing Statistics")
        print("=" * 60)
        print("No queries processed yet.")
        print("\nStart the bot and send some messages to see statistics.")
        print("=" * 60 + "\n")
        return

    print("\n📊 Routing Statistics")
    print("=" * 60)
    print(f"Total queries processed: {stats['total']}\n")

    # Sort by count
    sorted_intents = sorted(
        stats["distribution"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    max_count = max(stats["distribution"].values()) if stats["distribution"] else 1

    # Display distribution
    intent_labels = {
        "tool_only": "🔧 TOOL_ONLY      ",
        "rag_only": "📚 RAG_ONLY       ",
        "tool_rag": "🔧📚 TOOL_THEN_RAG",
        "chitchat": "💬 CHITCHAT       ",
    }

    for intent, count in sorted_intents:
        if count == 0:
            continue

        percentage = stats["percentages"][intent]
        label = intent_labels.get(intent, intent)
        bar = format_bar(count, max_count, width=30)

        print(f"{label}  {bar}  {count:3d} ({percentage:5.1f}%)")

    print("\n" + "=" * 60)

    # Insights
    print("\n💡 Insights:")

    tool_only_pct = stats["percentages"].get("tool_only", 0)
    chitchat_pct = stats["percentages"].get("chitchat", 0)
    rag_only_pct = stats["percentages"].get("rag_only", 0)

    fast_path = tool_only_pct + chitchat_pct

    print(f"  • Fast path (no RAG): {fast_path:.1f}%")
    print(f"  • Knowledge base used: {rag_only_pct + stats['percentages'].get('tool_rag', 0):.1f}%")

    if fast_path > 50:
        print("  ✅ Great! Over 50% queries use fast path (no RAG overhead)")
    elif fast_path > 30:
        print("  👍 Good fast path usage. Consider adding more tool patterns.")
    else:
        print("  ⚠️  Low fast path usage. Review patterns to reduce RAG calls.")

    print("\n" + "=" * 60 + "\n")


def main():
    """Main entry point."""
    if "--reset" in sys.argv:
        reset_routing_stats()
        print("✅ Routing statistics have been reset.\n")
        return

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return

    display_stats()


if __name__ == "__main__":
    main()
