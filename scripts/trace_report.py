"""Trace report: latency percentiles, token spend, cost, route mix.

Reads data/processed/traces.jsonl and prints the numbers the README needs:
p50/p95 latency, cost per query, stage breakdown, error counts.

Run: python -m scripts.trace_report
"""

import json
import sys
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.tracing import TRACE_PATH, percentile


def main() -> None:
    if not TRACE_PATH.exists():
        print(f"no traces yet at {TRACE_PATH} — run a chat or eval query first")
        return

    traces = [
        json.loads(line)
        for line in TRACE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not traces:
        print("traces file is empty")
        return

    durations = [trace["duration_ms"] for trace in traces]
    costs = [trace.get("cost_usd", 0.0) for trace in traces]
    errors = [trace for trace in traces if trace.get("error")]
    routes = Counter(trace.get("route", "?") for trace in traces)

    print(f"queries traced: {len(traces)} | errors: {len(errors)}")
    print(f"latency: p50 {percentile(durations, 0.5):,.0f} ms | p95 {percentile(durations, 0.95):,.0f} ms")
    print(f"tokens: {sum(trace.get('tokens_in', 0) for trace in traces):,} in / "
          f"{sum(trace.get('tokens_out', 0) for trace in traces):,} out")
    print(f"cost estimate: ${sum(costs):.6f} total | ${sum(costs) / len(traces):.6f} per query\n")

    print("route mix:")
    for route, count in routes.most_common():
        print(f"  {route}: {count}")

    print("\nby stage:")
    stages: dict[str, dict] = defaultdict(lambda: {"n": 0, "ms": 0, "in": 0, "out": 0})
    for trace in traces:
        for span in trace.get("spans", []):
            stats = stages[span["name"]]
            stats["n"] += 1
            stats["ms"] += span.get("duration_ms", 0)
            stats["in"] += span.get("tokens_in", 0)
            stats["out"] += span.get("tokens_out", 0)
    for name, stats in sorted(stages.items()):
        avg_ms = stats["ms"] / stats["n"]
        print(
            f"  {name:14} calls {stats['n']:4} | avg {avg_ms:7,.0f} ms | "
            f"tokens {stats['in']:8,} in / {stats['out']:8,} out"
        )


if __name__ == "__main__":
    main()
