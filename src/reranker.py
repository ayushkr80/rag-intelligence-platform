"""Cross-encoder reranker: scores (query, chunk) pairs for true relevance.

Unlike embeddings (which embed query and chunk separately), a cross-encoder
reads them together — slower, far more accurate. We rerank only the ~24
hybrid candidates, never the whole corpus.

torch can be blocked by OS application-control policies; callers must check
rerank_available() and degrade to plain hybrid ranking when False.
"""

try:
    from sentence_transformers import CrossEncoder

    _IMPORT_OK = True
except (ImportError, OSError):
    CrossEncoder = None
    _IMPORT_OK = False

from src import config

_model: "CrossEncoder | None" = None


def rerank_available() -> bool:
    """False when torch/sentence-transformers cannot load on this machine."""
    return _IMPORT_OK


def get_model() -> CrossEncoder:
    """Lazily load the reranker model once per process."""
    global _model
    if not _IMPORT_OK:
        raise RuntimeError("reranker unavailable: torch could not be loaded")
    if _model is None:
        _model = CrossEncoder(config.RERANKER_MODEL)
    return _model


def rerank(query: str, texts: list[str]) -> list[tuple[float, int]]:
    """Score each text against the query; return (score, position) best-first."""
    if not _IMPORT_OK:
        raise RuntimeError("reranker unavailable: torch could not be loaded")
    scores = get_model().predict([(query, text) for text in texts])
    order = sorted(range(len(texts)), key=lambda position: scores[position], reverse=True)
    return [(float(scores[position]), position) for position in order]
