"""Retrieval pipeline with switchable modes for the ablation study.

Modes:
- dense          — pure vector search (the Phase 1 baseline)
- hybrid         — dense + BM25 fused with RRF (wide recall, rank-blind order)
- hybrid_rerank  — hybrid candidates narrowed to ~24, then cross-encoder rerank
"""

import json

import numpy as np

from src import config
from src.bm25 import BM25Index
from src.hybrid import reciprocal_rank_fusion
from src.permissions import permitted_indices
from src.vector_store import InMemoryVectorStore

DENSE_CANDIDATES = 40
BM25_CANDIDATES = 60
RERANK_POOL = 24

_RERANK_WARNED = False


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
            {
                "company": chunk["company"],
                "year": chunk["year"],
                "page": chunk["page"],
                "level": chunk.get("level", "public"),
            }
            for chunk in chunks
        ],
    )
    bm25 = BM25Index()
    bm25.add([chunk["text"] for chunk in chunks])
    return store, bm25, chunks


def retrieve(
    query_vector: list[float],
    question: str,
    store: InMemoryVectorStore,
    bm25: BM25Index,
    mode: str,
    top_k: int = 4,
    role: str = "employee",
) -> list[dict]:
    """Return the top_k permitted chunk dicts for the requested mode.

    Permission filtering happens here — inside retrieval — over an
    over-fetched candidate pool, so restricted documents never consume
    result slots even when they rank highly.
    """
    allowed = permitted_indices(store.metadatas, role)

    if mode == "dense":
        hits = store.search(query_vector, top_k=top_k * 4)
        hits = [hit for hit in hits if hit[3] in allowed][:top_k]
        return [
            {"text": text, "metadata": meta, "index": index, "dense_score": score}
            for score, text, meta, index in hits
        ]

    dense_hits = [
        hit for hit in store.search(query_vector, top_k=DENSE_CANDIDATES) if hit[3] in allowed
    ]
    bm25_hits = [
        (score, index)
        for score, index in bm25.search(question, top_k=BM25_CANDIDATES)
        if index in allowed
    ]
    fused = reciprocal_rank_fusion(
        [index for _, _, _, index in dense_hits],
        [index for _, index in bm25_hits],
    )[:RERANK_POOL]

    if mode == "hybrid":
        return [
            {
                "text": store.texts[index],
                "metadata": store.metadatas[index],
                "index": index,
                "rrf_score": score,
            }
            for score, index in fused[:top_k]
        ]

    if mode == "hybrid_rerank":
        from src.reranker import rerank, rerank_available

        if not rerank_available():
            global _RERANK_WARNED
            if not _RERANK_WARNED:
                print(
                    "  (reranker unavailable on this machine — degrading to hybrid)",
                    flush=True,
                )
                _RERANK_WARNED = True
            return [
                {
                    "text": store.texts[index],
                    "metadata": store.metadatas[index],
                    "index": index,
                    "rrf_score": score,
                }
                for score, index in fused[:top_k]
            ]

        candidate_texts = [store.texts[index] for _, index in fused]
        reranked = rerank(question, candidate_texts)
        return [
            {
                "text": candidate_texts[position],
                "metadata": store.metadatas[fused[position][1]],
                "index": fused[position][1],
                "rerank_score": score,
            }
            for score, position in reranked[:top_k]
        ]

    raise ValueError(f"unknown retrieval mode: {mode}")
