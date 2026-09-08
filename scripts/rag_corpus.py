"""RAG over the real corpus: three 10-K filings, metadata-rich retrieval, cited answers.

Loads the persisted index (no re-embedding), retrieves with provenance, and
asks the generator to cite (company, year, page) — the citation format the
final product will expose in its UI.
"""

import json

import numpy as np

from src import config
from src.embeddings import embed_query
from src.generator import generate_answer
from src.vector_store import InMemoryVectorStore

QUESTIONS = [
    "How much revenue did Apple's Services segment generate in fiscal 2024?",
    "What was Microsoft's total revenue in fiscal year 2024?",
    "What was NVIDIA's Data Center revenue in fiscal 2025 and how did it change?",
    "Which company grew its cloud or data center business fastest?",
    "How much dividend per share did Apple pay in 2024?",
]


def load_index() -> InMemoryVectorStore:
    chunks = json.loads(
        (config.PROCESSED_DATA_DIR / "index_chunks.json").read_text(encoding="utf-8")
    )
    vectors = np.load(config.PROCESSED_DATA_DIR / "index_vectors.npy")
    store = InMemoryVectorStore()
    store.add(
        [chunk["text"] for chunk in chunks],
        vectors.tolist(),
        [
            {"company": chunk["company"], "year": chunk["year"], "page": chunk["page"]}
            for chunk in chunks
        ],
    )
    return store


def main() -> None:
    store = load_index()

    for question in QUESTIONS:
        print("=" * 78)
        print("QUESTION:", question)

        results = store.search(embed_query(question), top_k=4)
        for score, text, meta, _ in results:
            preview = text[:60].replace("\n", " ")
            print(f"  {score:.3f}  [{meta['company']} p.{meta['page']}]  {preview}...")

        context = "\n\n".join(
            f"[{i}] ({meta['company']} {meta['year']} 10-K, page {meta['page']})\n{text}"
            for i, (_, text, meta, _) in enumerate(results)
        )
        print("\nANSWER:", generate_answer(question, context))
        print()


if __name__ == "__main__":
    main()
