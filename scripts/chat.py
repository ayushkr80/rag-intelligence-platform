"""Interactive chat over the filings — the project's first product surface.

Pipeline per turn: rewrite (standalone query) -> hybrid retrieval -> rerank ->
grounded answer with citations. Type 'exit' to quit.

Run: .venv\\Scripts\\python -m scripts.chat
"""

from src import config
from src.embeddings import embed_query_cached
from src.generator import generate_answer
from src.query_rewriter import rewrite_query
from src.retrieval import load_corpus, retrieve


def main() -> None:
    store, bm25, chunks = load_corpus()
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

        search_query = rewrite_query(history, question)
        if search_query != question:
            print(f"  (searching: {search_query})")

        results = retrieve(
            embed_query_cached(search_query), search_query, store, bm25, config.RAG_MODE
        )
        context = "\n\n".join(
            f"[{i}] ({chunk['metadata']['company']} {chunk['metadata']['year']} 10-K, "
            f"page {chunk['metadata']['page']})\n{chunk['text']}"
            for i, chunk in enumerate(results)
        )
        answer = generate_answer(question, context)
        print(f"\nassistant> {answer}\n")
        history.append((question, answer))


if __name__ == "__main__":
    main()
