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
from src.vector_store import InMemoryVectorStore

DENSE_CANDIDATES = 20
BM25_CANDIDATES = 20
RERANK_POOL = 24


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


def retrieve(
    query_vector: list[float],
    question: str,
    store: InMemoryVectorStore,
    bm25: BM25Index,
    mode: str,
    top_k: int = 4,
) -> list[dict]:
    """Return the top_k chunk dicts for the requested retrieval mode."""
    if mode == "dense":
        hits = store.search(query_vector, top_k=top_k)
        return [
            {"text": text, "metadata": meta, "index": index, "dense_score": score}
            for score, text, meta, index in hits
        ]

    dense_hits = store.search(query_vector, top_k=DENSE_CANDIDATES)
    bm25_hits = bm25.search(question, top_k=BM25_CANDIDATES)
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
        from src.reranker import rerank

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
