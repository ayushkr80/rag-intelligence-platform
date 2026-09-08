"""Compare dense-only vs BM25-only vs hybrid retrieval on the hard questions.

No generation calls — pure retrieval comparison, so the improvement we measure
is attributable to the retriever alone.
"""

import json

import numpy as np

from src import config
from src.bm25 import BM25Index
from src.embeddings import embed_query
from src.hybrid import reciprocal_rank_fusion
from src.vector_store import InMemoryVectorStore

QUESTIONS = [
    "How much revenue did Apple's Services segment generate in fiscal 2024?",
    "What was Microsoft's total revenue in fiscal year 2024?",
    "What was NVIDIA's Data Center revenue in fiscal 2025?",
    "How much dividend per share did Apple pay in 2024?",
    "How many shares did Apple repurchase in fiscal 2024?",
]

TOP_K = 4


def load_corpus() -> tuple[InMemoryVectorStore, BM25Index, list[dict]]:
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
    bm25 = BM25Index()
    bm25.add([chunk["text"] for chunk in chunks])
    return store, bm25, chunks


def preview(chunks: list[dict], index: int) -> str:
    chunk = chunks[index]
    text = chunk["text"][:55].replace("\n", " ")
    return f"[{chunk['company'][:5]} p.{chunk['page']}] {text}"


def main() -> None:
    store, bm25, chunks = load_corpus()

    for question in QUESTIONS:
        print("=" * 78)
        print("QUESTION:", question)

        query_vector = embed_query(question)
        dense = store.search(query_vector, top_k=TOP_K)
        dense_indices = []
        for score, text, meta, index in dense:
            dense_indices.append(index)
            print(f"  dense  {score:.3f}  {preview(chunks, index)}")

        bm25_results = bm25.search(question, top_k=TOP_K)
        bm25_indices = []
        for score, index in bm25_results:
            bm25_indices.append(index)
            print(f"  bm25   {score:7.2f}  {preview(chunks, index)}")

        print("  --- hybrid (RRF of dense + bm25) ---")
        for score, index in reciprocal_rank_fusion(dense_indices, bm25_indices)[:TOP_K]:
            print(f"  hybrid {score:.4f}  {preview(chunks, index)}")
        print()


if __name__ == "__main__":
    main()
