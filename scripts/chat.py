"""Interactive chat over the filings — the project's first product surface.

Pipeline per turn: rewrite (standalone query) -> route (vector or sql) ->
tool -> grounded answer with citations. Type 'exit' to quit.

Run: .venv\\Scripts\\python -m scripts.chat
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent import answer


def main() -> None:
    role = input("role [analyst/employee, default employee]> ").strip() or "employee"
    print(f"chatting as role: {role}\n")
    history: list[tuple[str, str]] = []
    print("Chat with the 10-K corpus (Apple, Microsoft, NVIDIA). Type 'exit' to quit.\n")

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        result = answer(history, question, role=role)
        if result["search_query"] != question:
            print(f"  (searching: {result['search_query']})")
        print(f"  (route: {result['route']})")
        print(f"\nassistant> {result['answer']}\n")
        history.append((question, result["answer"]))


if __name__ == "__main__":
    main()
