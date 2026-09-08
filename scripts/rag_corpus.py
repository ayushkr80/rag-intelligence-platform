"""RAG over the real corpus using the full retrieval pipeline.

RAG_MODE env var switches the retriever (dense / hybrid / hybrid_rerank) so
answers can be compared across architectures on the same questions.
"""

from src import config
from src.embeddings import embed_query_cached
from src.generator import generate_answer
from src.retrieval import load_corpus, retrieve

QUESTIONS = [
    "How much revenue did Apple's Services segment generate in fiscal 2024?",
    "What was Microsoft's total revenue in fiscal year 2024?",
    "What was NVIDIA's Data Center revenue in fiscal 2025 and how did it change?",
    "Which company grew its cloud or data center business fastest?",
    "How much dividend per share did Apple pay in 2024?",
]


def main() -> None:
    store, bm25, chunks = load_corpus()

    for question in QUESTIONS:
        print("=" * 78)
        print("QUESTION:", question)

        query_vector = embed_query_cached(question)
        results = retrieve(query_vector, question, store, bm25, config.RAG_MODE)
        for chunk in results:
            meta = chunk["metadata"]
            preview = chunk["text"][:60].replace("\n", " ")
            print(f"  [{meta['company']} p.{meta['page']}]  {preview}...")

        context = "\n\n".join(
            f"[{i}] ({chunk['metadata']['company']} {chunk['metadata']['year']} 10-K, "
            f"page {chunk['metadata']['page']})\n{chunk['text']}"
            for i, chunk in enumerate(results)
        )
        print("\nANSWER:", generate_answer(question, context))
        print()


if __name__ == "__main__":
    main()
