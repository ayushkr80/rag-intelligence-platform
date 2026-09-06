"""Embedding client: turns text into vectors so similar meanings land close together."""

from google import genai

from src import config
from src.retry import call_with_backoff

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Lazily create one shared Gemini client for the whole process."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.active_api_key())
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, one 3072-dim vector per text, order preserved."""
    result = call_with_backoff(
        lambda: get_client().models.embed_content(
            model=config.GEMINI_EMBEDDING_MODEL,
            contents=texts,
        ),
        label="embed",
    )
    return [item.values for item in result.embeddings]


def embed_query(text: str) -> list[float]:
    """Embed a single query."""
    return embed_texts([text])[0]
