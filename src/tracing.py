"""Lightweight tracing: per-query spans with latency, tokens, and cost.

No external dependencies — spans are plain dicts, persisted as JSON lines.
Costs are ESTIMATES from a configurable per-1M-token price table; free-tier
calls still record tokens, so cost reporting is ready when billing turns on.
"""

import json
import time

from src import config

TRACE_PATH = config.PROCESSED_DATA_DIR / "traces.jsonl"

# USD per 1M tokens (input, output) — update with current published pricing.
MODEL_COSTS = {
    "gemini-3.8-flash": (0.15, 0.60),
    "gemini-3.7-flash": (0.15, 0.60),
    "gemini-3.6-flash": (0.15, 0.60),
    "gemini-3.5-flash": (0.15, 0.60),
    "gemini-3.1-flash-lite": (0.05, 0.20),
    "gemini-embedding-001": (0.0, 0.0),
}
DEFAULT_COSTS = (0.0, 0.0)


def begin_query(question: str, role: str) -> dict:
    """Open a trace for one user query."""
    return {"question": question, "role": role, "started": time.time(), "spans": []}


def add_span(
    trace: dict,
    name: str,
    model: str = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    duration_ms: int = 0,
    **meta,
) -> dict:
    """Record one pipeline step (LLM call or local stage) into the trace."""
    span = {
        "name": name,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "duration_ms": duration_ms,
        **meta,
    }
    trace["spans"].append(span)
    return span


def span_cost(span: dict) -> float:
    price_in, price_out = MODEL_COSTS.get(span.get("model") or "", DEFAULT_COSTS)
    return span.get("tokens_in", 0) / 1e6 * price_in + span.get("tokens_out", 0) / 1e6 * price_out


def end_query(trace: dict, error: str = None) -> dict:
    """Finalize a trace: totals, cost, persist one JSON line."""
    trace["duration_ms"] = round((time.time() - trace["started"]) * 1000)
    trace["error"] = error
    trace["tokens_in"] = sum(span.get("tokens_in", 0) for span in trace["spans"])
    trace["tokens_out"] = sum(span.get("tokens_out", 0) for span in trace["spans"])
    trace["cost_usd"] = round(sum(span_cost(span) for span in trace["spans"]), 6)
    trace.pop("started", None)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return trace


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty list."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]
