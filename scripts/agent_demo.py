"""Demo: the agentic router in action on the five hard questions.

Watch the route line — numeric questions go to SQL, everything else to
hybrid vector search. Same pipeline, different tools, one interface.
Optionally pass 1-based question indices: python -m scripts.agent_demo 3 4
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent import answer

QUESTIONS = [
    "How much revenue did Apple's Services segment generate in fiscal 2024?",
    "What was Microsoft's total revenue in fiscal year 2024?",
    "Which company grew its cloud or data center business fastest?",
    "How much dividend per share did Apple pay in 2024?",
    "Why did NVIDIA's revenue grow so much in fiscal 2025?",
]


def main() -> None:
    wanted = {int(arg) - 1 for arg in sys.argv[1:]} or set(range(len(QUESTIONS)))
    history: list[tuple[str, str]] = []
    for index in sorted(wanted):
        question = QUESTIONS[index]
        print("=" * 78)
        print("QUESTION:", question)
        result = answer(history, question)
        print(f"  route: {result['route']}")
        if result["search_query"] != question:
            print(f"  search: {result['search_query']}")
        if result["sql_debug"]:
            print(f"  sql: {result['sql_debug']['sql']}")
            print(f"  rows: {result['sql_debug']['rows'][:8]}")
        print(f"\nANSWER: {result['answer']}\n")
        history.append((question, result["answer"]))
        time.sleep(1.5)


if __name__ == "__main__":
    main()
