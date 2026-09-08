"""Cross-encoder reranker: scores (query, chunk) pairs for true relevance.

Unlike embeddings (which embed query and chunk separately), a cross-encoder
reads them together — slower, far more accurate. We rerank only the ~24
hybrid candidates, never the whole corpus.
"""

from sentence_transformers import CrossEncoder

from src import config

_model: CrossEncoder | None = None


def get_model() -> CrossEncoder:
    """Lazily load the reranker model once per process."""
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANKER_MODEL)
    return _model


def rerank(query: str, texts: list[str]) -> list[tuple[float, int]]:
    """Score each text against the query; return (score, position) best-first."""
    scores = get_model().predict([(query, text) for text in texts])
    order = sorted(range(len(texts)), key=lambda position: scores[position], reverse=True)
    return [(float(scores[position]), position) for position in order]
