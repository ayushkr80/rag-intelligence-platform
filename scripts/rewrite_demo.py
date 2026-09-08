"""Scripted multi-turn demo: see what query rewriting changes, turn by turn.

Each turn prints the rewritten search query and the cited answer. Follow-ups
are meaningless alone ("What about Microsoft?") — watch the rewriter make
them retrievable.
"""

from src.embeddings import embed_query_cached
from src.generator import generate_answer
from src.query_rewriter import rewrite_query
from src.retrieval import load_corpus, retrieve

CONVERSATION = [
    "How much dividend per share did Apple pay in 2024?",
    "What about Microsoft?",
    "And NVIDIA?",
]


def main() -> None:
    store, bm25, chunks = load_corpus()
    history: list[tuple[str, str]] = []

    for question in CONVERSATION:
        print("=" * 78)
        print("USER:", question)

        search_query = rewrite_query(history, question)
        print(f"  (searching: {search_query!r})")

        results = retrieve(
            embed_query_cached(search_query), search_query, store, bm25, "hybrid_rerank"
        )
        for chunk in results[:2]:
            meta = chunk["metadata"]
            preview = chunk["text"][:60].replace("\n", " ")
            print(f"  [{meta['company']} p.{meta['page']}]  {preview}...")

        context = "\n\n".join(
            f"[{i}] ({chunk['metadata']['company']} {chunk['metadata']['year']} 10-K, "
            f"page {chunk['metadata']['page']})\n{chunk['text']}"
            for i, chunk in enumerate(results)
        )
        answer = generate_answer(question, context)
        print("\nASSISTANT:", answer, "\n")
        history.append((question, answer))


if __name__ == "__main__":
    main()
