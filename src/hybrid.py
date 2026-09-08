"""Hybrid retrieval: fuse dense (meaning) and BM25 (keyword) rankings with RRF.

Reciprocal Rank Fusion: a document's fused score is the sum of 1/(k + rank)
across each ranking it appears in. Rank-based, so incomparable score scales
(cosine 0.0-1.0 vs BM25 0-30+) never mix.
"""

K = 60


def reciprocal_rank_fusion(*ranked_lists: list[int]) -> list[tuple[float, int]]:
    """Fuse ranked lists of indices into one ranking, best first."""
    fused: dict[int, float] = {}
    for results in ranked_lists:
        for rank, index in enumerate(results):
            fused[index] = fused.get(index, 0.0) + 1.0 / (K + rank + 1)
    return sorted(
        ((score, index) for index, score in fused.items()),
        reverse=True,
    )
