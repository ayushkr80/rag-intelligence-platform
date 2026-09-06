"""Shared retry helper for transient LLM API failures (429 rate limit, 5xx)."""

import random
import time

from google.genai import errors as genai_errors

MAX_ATTEMPTS = 6
INITIAL_BACKOFF_SECONDS = 2.0
TRANSIENT_CODES = ("429", "500", "503")


def call_with_backoff(call, label: str):
    """Run the call, retrying transient API errors with exponential backoff."""
    delay = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call()
        except genai_errors.APIError as exc:
            transient = any(code in str(exc) for code in TRANSIENT_CODES)
            if not transient or attempt == MAX_ATTEMPTS:
                raise
            wait = delay + random.uniform(0, 1)
            print(f"  {label}: transient API error — retry {attempt} in {wait:.1f}s", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")
