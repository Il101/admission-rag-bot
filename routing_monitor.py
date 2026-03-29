"""
Live routing monitor - tail bot logs and display routing decisions in real-time.

Shows classification decisions as they happen with color-coded output.

Usage:
    python3 routing_monitor.py [log_file]

If log_file is not specified, tries to find bot log automatically.
"""

import sys
import re
import time
from pathlib import Path
from datetime import datetime


# ANSI color codes
class Colors:
    TOOL_ONLY = '\033[92m'    # Green
    RAG_ONLY = '\033[94m'     # Blue
    TOOL_RAG = '\033[93m'     # Yellow
    CHITCHAT = '\033[95m'     # Magenta
    ERROR = '\033[91m'        # Red
    RESET = '\033[0m'         # Reset
    BOLD = '\033[1m'          # Bold
    DIM = '\033[2m'           # Dim


def colorize_intent(intent: str) -> str:
    """Add color to intent string."""
    color_map = {
        "tool_only": Colors.TOOL_ONLY,
        "rag_only": Colors.RAG_ONLY,
        "tool_rag": Colors.TOOL_RAG,
        "chitchat": Colors.CHITCHAT,
    }
    color = color_map.get(intent, Colors.RESET)
    return f"{color}{intent.upper()}{Colors.RESET}"


def parse_routing_log(line: str) -> dict:
    """Parse routing log line and extract info.

    Expected format:
    [ROUTING] user=12345 intent=tool_only confidence=0.92 tools=['get_my_progress'] ...
    """
    pattern = r"\[ROUTING\] user=(\d+) intent=(\w+) confidence=([\d.]+) tools=\[(.*?)\] (?:latency=([\d.]+)ms )?reason=(.*)"
    match = re.search(pattern, line)

    if not match:
        return None

    user_id, intent, confidence, tools_str, latency, reason = match.groups()

    # Parse tools list
    tools = []
    if tools_str:
        tools = [t.strip().strip("'\"") for t in tools_str.split(",")]

    return {
        "user_id": int(user_id),
        "intent": intent,
        "confidence": float(confidence),
        "tools": tools,
        "latency_ms": float(latency) if latency else 0.0,
        "reason": reason.strip(),
        "timestamp": datetime.now(),
    }


def format_routing_info(info: dict) -> str:
    """Format routing info for display."""
    ts = info["timestamp"].strftime("%H:%M:%S")
    user = info["user_id"]
    intent = colorize_intent(info["intent"])
    confidence = info["confidence"]

    # Confidence indicator
    if confidence >= 0.9:
        conf_icon = "🎯"
    elif confidence >= 0.7:
        conf_icon = "✅"
    else:
        conf_icon = "⚠️"

    tools_str = ""
    if info["tools"]:
        tools_str = f" → {', '.join(info['tools'])}"

    latency_str = ""
    if info["latency_ms"] > 0:
        latency_str = f" ({info['latency_ms']:.1f}ms)"

    line = f"{Colors.DIM}[{ts}]{Colors.RESET} {conf_icon} User {user}: {intent} (conf={confidence:.2f}){tools_str}{latency_str}"

    # Add reason on next line if interesting
    if info["reason"] and "точно классифицировать" in info["reason"]:
        line += f"\n  {Colors.DIM}└─ {info['reason']}{Colors.RESET}"

    return line


def tail_file(filename: str):
    """Tail a file and yield new lines."""
    try:
        with open(filename, 'r') as f:
            # Go to end of file
            f.seek(0, 2)

            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n{Colors.DIM}Monitoring stopped.{Colors.RESET}")
    except FileNotFoundError:
        print(f"{Colors.ERROR}Error: Log file not found: {filename}{Colors.RESET}")
        sys.exit(1)


def find_log_file() -> str:
    """Try to find bot log file automatically."""
    possible_paths = [
        "bot.log",
        "logs/bot.log",
        "/tmp/bot.log",
        "../logs/bot.log",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return path

    return None


def main():
    """Main entry point."""
    print(f"\n{Colors.BOLD}🔍 Live Routing Monitor{Colors.RESET}")
    print(f"{Colors.DIM}{'=' * 60}{Colors.RESET}\n")

    # Get log file
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = find_log_file()
        if not log_file:
            print(f"{Colors.ERROR}Error: Could not find log file.{Colors.RESET}")
            print(f"\nUsage: {sys.argv[0]} [log_file]")
            print(f"\nLooking for bot log at:")
            print("  - bot.log")
            print("  - logs/bot.log")
            print("  - /tmp/bot.log")
            sys.exit(1)

    print(f"Monitoring: {Colors.BOLD}{log_file}{Colors.RESET}")
    print(f"Press Ctrl+C to stop\n")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

    # Track stats
    stats = {
        "tool_only": 0,
        "rag_only": 0,
        "tool_rag": 0,
        "chitchat": 0,
        "total": 0,
    }

    try:
        for line in tail_file(log_file):
            if "[ROUTING]" in line:
                info = parse_routing_log(line)
                if info:
                    # Update stats
                    stats["total"] += 1
                    if info["intent"] in stats:
                        stats[info["intent"]] += 1

                    # Display
                    print(format_routing_info(info))

                    # Show stats every 10 decisions
                    if stats["total"] % 10 == 0:
                        print(f"\n{Colors.DIM}Stats: {stats['total']} total | "
                              f"TOOL:{stats['tool_only']} RAG:{stats['rag_only']} "
                              f"BOTH:{stats['tool_rag']} CHAT:{stats['chitchat']}{Colors.RESET}\n")

    except KeyboardInterrupt:
        print(f"\n\n{Colors.DIM}{'─' * 60}{Colors.RESET}")
        print(f"\n{Colors.BOLD}Final Statistics:{Colors.RESET}")
        for key, value in stats.items():
            if key != "total":
                pct = (value / stats["total"] * 100) if stats["total"] > 0 else 0
                print(f"  {key}: {value} ({pct:.1f}%)")
        print(f"\n  Total: {stats['total']}")
        print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
