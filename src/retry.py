"""Shared retry helper for transient LLM API failures (429 rate limit, 5xx)."""

import random
import time

from google.genai import errors as genai_errors

MAX_ATTEMPTS = 6
INITIAL_BACKOFF_SECONDS = 2.0
TRANSIENT_CODES = ("429", "500", "503")

_drained_models: set[str] = set()


def _is_daily_quota(exc: Exception) -> bool:
    """Daily-quota 429s (quota id contains 'PerDay') cannot be retried away."""
    return "429" in str(exc) and "PerDay" in str(exc)


def call_with_backoff(call, label: str):
    """Run the call, retrying transient API errors with exponential backoff."""
    delay = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call()
        except genai_errors.APIError as exc:
            if _is_daily_quota(exc):
                raise
            transient = any(code in str(exc) for code in TRANSIENT_CODES)
            if not transient or attempt == MAX_ATTEMPTS:
                raise
            wait = delay + random.uniform(0, 1)
            print(f"  {label}: transient API error — retry {attempt} in {wait:.1f}s", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def call_with_failover(models: list[str], make_call, label: str):
    """Try each model in order; a model whose daily quota is drained trips a
    circuit breaker and is skipped instantly for the rest of the process."""
    ordered = [m for m in dict.fromkeys(models) if m not in _drained_models]
    if not ordered:
        raise RuntimeError(
            "all failover models have drained their daily quota — rerun after the "
            "daily reset (midnight Pacific) or add models to FALLBACK_MODELS"
        )
    last_error: Exception | None = None
    for position, model in enumerate(ordered):
        try:
            return call_with_backoff(lambda: make_call(model), label=f"{label}[{model}]")
        except genai_errors.APIError as exc:
            last_error = exc
            if _is_daily_quota(exc):
                _drained_models.add(model)
                if position < len(ordered) - 1:
                    print(f"  {model} daily quota drained — failing over", flush=True)
    raise last_error
