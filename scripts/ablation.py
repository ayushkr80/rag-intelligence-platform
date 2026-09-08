"""Ablation study: retrieval quality across dense / hybrid / hybrid_rerank.

One query embedding per question (cached), reused across all three modes, so
the comparison isolates the retriever — and the API cost is 5 texts total.
"""

from src.embeddings import embed_query_cached
from src.retrieval import load_corpus, retrieve

QUESTIONS = [
    "How much revenue did Apple's Services segment generate in fiscal 2024?",
    "What was Microsoft's total revenue in fiscal year 2024?",
    "What was NVIDIA's Data Center revenue in fiscal 2025?",
    "How much dividend per share did Apple pay in 2024?",
    "How many shares did Apple repurchase in fiscal 2024?",
]

MODES = ("dense", "hybrid", "hybrid_rerank")


def preview(chunk: dict) -> str:
    meta = chunk["metadata"]
    text = chunk["text"][:55].replace("\n", " ")
    return f"[{meta['company'][:5]} p.{meta['page']}] {text}"


def score_of(chunk: dict) -> float:
    return next(value for key, value in chunk.items() if key.endswith("_score"))


def main() -> None:
    store, bm25, chunks = load_corpus()
    for question in QUESTIONS:
        print("=" * 78)
        print("QUESTION:", question)
        query_vector = embed_query_cached(question)
        for mode in MODES:
            print(f"  --- {mode} ---")
            for chunk in retrieve(query_vector, question, store, bm25, mode):
                print(f"  {score_of(chunk):8.4f}  {preview(chunk)}")
        print()


if __name__ == "__main__":
    main()
