"""Embedding client: turns text into vectors so similar meanings land close together."""

import hashlib
import json
import time

from google import genai

from src import config
from src.retry import call_with_backoff
from src.tracing import add_span

_client: genai.Client | None = None

QUERY_CACHE_PATH = config.DATA_DIR / "eval_cache" / "query_embeddings.json"


def get_client() -> genai.Client:
    """Lazily create one shared Gemini client for the whole process."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.active_api_key())
    return _client


def embed_texts(texts: list[str], trace: dict = None, label: str = "embed") -> list[list[float]]:
    """Embed a batch of texts, one 3072-dim vector per text, order preserved."""
    start = time.perf_counter()
    result = call_with_backoff(
        lambda: get_client().models.embed_content(
            model=config.GEMINI_EMBEDDING_MODEL,
            contents=texts,
        ),
        label=label,
    )
    if trace is not None:
        usage = getattr(result, "usage_metadata", None)
        add_span(
            trace,
            label,
            model=config.GEMINI_EMBEDDING_MODEL,
            tokens_in=getattr(usage, "total_token_count", 0) or 0,
            tokens_out=0,
            duration_ms=round((time.perf_counter() - start) * 1000),
        )
    return [item.values for item in result.embeddings]


def embed_query(text: str) -> list[float]:
    """Embed a single query."""
    return embed_texts([text])[0]


def embed_query_cached(text: str, trace: dict = None) -> list[float]:
    """Embed a query once, then serve repeats from a local cache — queries
    repeat constantly across eval runs, and every avoidable API call is quota."""
    QUERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if QUERY_CACHE_PATH.exists():
        cache = json.loads(QUERY_CACHE_PATH.read_text(encoding="utf-8"))
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    if key in cache:
        if trace is not None:
            add_span(trace, "embed-query", model="cache", tokens_in=0, tokens_out=0)
        return cache[key]
    vector = embed_query(text)
    cache[key] = vector
    QUERY_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    return vector
