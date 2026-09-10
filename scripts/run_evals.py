"""Eval runner: deterministic retrieval ablation, optional judge scoring, gate.

Modes of operation:
- default:          retrieval eval across modes (zero LLM calls; embeddings cached)
- --agent:          run full agent answers + deterministic grading (costs quota)
- --judge:          additionally score answers with the LLM judge (costs quota)
- --gate:           regression gate — fail (exit 1) if hit rate drops below baseline
- --update-baseline: save current retrieval metrics as the new baseline
- --limit N:        cap questions (for cheap smoke runs)

The regression gate makes evals a safety net: any code change that drops
retrieval quality fails the check before it can ship.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config
from src.agent import answer
from src.evals import answer_matches, judge_answer, source_hit
from src.embeddings import embed_query_cached
from src.retrieval import load_corpus, retrieve

MODES = ("dense", "hybrid", "hybrid_rerank")
GOLDEN_PATH = config.EVALS_DIR / "golden_set.json"
BASELINE_PATH = config.EVALS_DIR / "baseline.json"


def load_golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"]


def retrieval_eval(questions: list[dict], modes: list[str]) -> list[dict]:
    """Score source hits and permission compliance per question per mode."""
    store, bm25, chunks = load_corpus()
    rows: list[dict] = []
    for question in questions:
        try:
            query_vector = embed_query_cached(question["question"])
        except Exception as exc:
            print(f"  skip (embed failed): {question['question'][:45]} — {exc}", flush=True)
            continue
        for mode in modes:
            results = retrieve(
                query_vector,
                question["question"],
                store,
                bm25,
                mode,
                role=question.get("role", "employee"),
            )
            leaked = any(
                chunk["metadata"].get("level", "public") != "public" for chunk in results
            )
            rows.append(
                {
                    "id": question["id"],
                    "category": question.get("category", "general"),
                    "mode": mode,
                    "source_hit": source_hit(results, question.get("source_hints", [])),
                    "permission_ok": question.get("role", "employee") == "employee"
                    or not leaked,
                }
            )
    return rows


def summarize(rows: list[dict], modes: list[str]) -> dict:
    summary = {}
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        perm_rows = [row for row in mode_rows if row["category"] == "permission"]
        summary[mode] = {
            "n": len(mode_rows),
            "source_hit_rate": round(
                sum(row["source_hit"] for row in mode_rows) / max(len(mode_rows), 1), 3
            ),
            "permission_leaks": sum(not row["permission_ok"] for row in perm_rows),
        }
    return summary


def print_table(summary: dict) -> None:
    print("\n| mode | evaluated | source hit@4 | permission leaks |")
    print("|---|---|---|---|")
    for mode, stats in summary.items():
        print(
            f"| {mode} | {stats['n']} | {stats['source_hit_rate']:.1%} | {stats['permission_leaks']} |"
        )
    print()


def agent_eval(questions: list[dict], limit: int, use_judge: bool) -> list[dict]:
    """Run full agent answers, grade deterministically, optionally judge.

    Crash-tolerant: per-question errors are recorded, not fatal, and rows
    persist incrementally so an interrupted run keeps its completed work.
    """
    results_path = config.EVALS_DIR / "agent_eval_results.json"
    previous: dict[int, dict] = {}
    if results_path.exists():
        previous = {
            row["id"]: row
            for row in json.loads(results_path.read_text(encoding="utf-8"))
        }
    rows: list[dict] = []
    for question in questions[:limit]:
        prior = previous.get(question["id"])
        if prior and "error" not in prior:
            rows.append(prior)
            print(f"  Q{question['id']}: cached", flush=True)
            continue
        print(f"  Q{question['id']}: {question['question'][:60]}", flush=True)
        try:
            result = answer([], question["question"], role=question.get("role", "employee"))
            deterministic = answer_matches(
                result["answer"],
                question.get("expected_facts", []),
                question.get("expect_refusal", False),
            )
            row = {
                "id": question["id"],
                "route": result["route"],
                "expected_route": question.get("expected_route"),
                "deterministic_pass": deterministic,
                "answer": result["answer"],
            }
            if use_judge:
                row["judge"] = judge_answer(
                    question["question"],
                    result["answer"],
                    question.get("expected_facts", []),
                    question.get("expect_refusal", False),
                )
                time.sleep(1.0)
            print(
                f"    route={row['route']} pass={deterministic} judge={row.get('judge', '-')}",
                flush=True,
            )
        except Exception as exc:
            row = {"id": question["id"], "error": str(exc)[:200]}
            print(f"    ERROR: {row['error']}", flush=True)
            time.sleep(3)
        rows.append(row)
        results_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        time.sleep(1.5)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="run full agent answers")
    parser.add_argument("--judge", action="store_true", help="score answers with LLM judge")
    parser.add_argument("--gate", action="store_true", help="regression gate vs baseline")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    questions = load_golden()
    rows = retrieval_eval(questions, list(MODES))
    summary = summarize(rows, list(MODES))
    print_table(summary)

    if args.agent or args.judge:
        agent_rows = agent_eval(questions, args.limit, use_judge=args.judge)
        graded = [row for row in agent_rows if row.get("deterministic_pass")]
        print(f"\nagent deterministic pass rate: {len(graded)}/{len(agent_rows)}")
        judged = [row for row in agent_rows if "judge" in row and row["judge"]["correctness"] >= 0]
        if judged:
            avg_correct = sum(row["judge"]["correctness"] for row in judged) / len(judged)
            avg_faithful = sum(row["judge"]["faithfulness"] for row in judged) / len(judged)
            print(f"judge averages: correctness {avg_correct:.2f}/2, faithfulness {avg_faithful:.2f}/2")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = config.EVALS_DIR / f"results_{stamp}.json"
    results_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )
    print(f"results saved -> {results_path.name}")

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"baseline updated -> {BASELINE_PATH.name}")

    if args.gate:
        if not BASELINE_PATH.exists():
            print("gate: no baseline found — run with --update-baseline first")
            sys.exit(2)
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        failed = False
        for mode, stats in summary.items():
            base_rate = baseline.get(mode, {}).get("source_hit_rate", 0)
            if stats["source_hit_rate"] < base_rate - 0.02:
                print(
                    f"REGRESSION: {mode} hit rate {stats['source_hit_rate']:.1%} "
                    f"< baseline {base_rate:.1%}"
                )
                failed = True
        print("gate: " + ("FAILED" if failed else "PASSED"))
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
