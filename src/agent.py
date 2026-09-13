"""Agent orchestrator: rewrite -> route -> tool -> grounded answer.

The full pipeline. Every question gets the cheapest sufficient tool; each
tool returns evidence in the same format, so generation stays uniform.

Phase 7: every query produces a trace (spans, latency, tokens, cost) that is
persisted to data/processed/traces.jsonl — including failures. Edge cases
handled: empty permitted result sets short-circuit to a free deterministic
refusal, and pipeline errors are recorded in the trace before re-raising.
"""

import time

from src import config
from src.embeddings import embed_query_cached
from src.generator import generate_answer
from src.graph_tool import query as query_graph
from src.guards import redact_injection, wrap_document
from src.query_rewriter import rewrite_query
from src.retrieval import load_corpus, retrieve
from src.router import route
from src.sql_tool import run_sql
from src.tracing import begin_query, end_query

_store = _bm25 = _chunks = None


def _corpus():
    global _store, _bm25, _chunks
    if _store is None:
        _store, _bm25, _chunks = load_corpus()
    return _store, _bm25, _chunks


def answer(history: list[tuple[str, str]], question: str, role: str = "employee") -> dict:
    """Run the full pipeline; returns route, search query, answer, and trace."""
    trace = begin_query(question, role)
    sql_debug = None
    graph_debug = None
    evidence = None
    try:
        search_query = rewrite_query(history, question, trace=trace)
        tool = route(search_query, trace=trace)
        trace["search_query"] = search_query
        trace["route"] = tool
        time.sleep(1.0)

        if tool == "sql":
            result = run_sql(search_query, trace=trace)
            if "error" in result:
                tool = "vector (sql blocked)"
            else:
                rows_text = "\n".join(str(row) for row in result["rows"][:20])
                evidence = (
                    f"SQL query executed:\n{result['sql']}\n\nRows (company, metric, segment, "
                    "fiscal_year, value, unit, source_page):\n"
                    f"{rows_text}\n\nUse these rows as the source of truth for numbers; "
                    "compute any rates or comparisons from them."
                )
                sql_debug = {"sql": result["sql"], "rows": result["rows"]}

        elif tool == "graph":
            graph_result = query_graph(search_query)
            if not graph_result["triples"]:
                tool = "vector (no graph matches)"
            else:
                facts = "\n".join(
                    f"{subject} | {relation} | {obj} | (page {page})"
                    for subject, relation, obj, page in graph_result["triples"]
                )
                evidence = (
                    "Knowledge graph facts (subject | relation | object | source page):\n"
                    f"{facts}\n\nUse these facts to answer; follow the connections "
                    "between them for multi-hop reasoning."
                )
                graph_debug = graph_result

        if tool != "sql" and evidence is None:
            store, bm25, _ = _corpus()
            results = retrieve(
                embed_query_cached(search_query, trace=trace),
                search_query,
                store,
                bm25,
                config.RAG_MODE,
                top_k=6,
                role=role,
            )
            if not results:
                trace["route"] = tool = f"{tool} (no permitted results)"
                end_query(trace, error=None)
                return {
                    "route": tool,
                    "search_query": search_query,
                    "sql_debug": None,
                    "graph_debug": None,
                    "answer": "Not in the documents.",
                    "trace": trace,
                }
            guarded = []
            redacted_total = 0
            for i, chunk in enumerate(results):
                clean_text, removed = redact_injection(chunk["text"])
                redacted_total += removed
                tag = (
                    f"[{i}] ({chunk['metadata']['company']} {chunk['metadata']['year']} "
                    f"10-K, page {chunk['metadata']['page']}; {removed} injected line(s) removed)"
                    if removed
                    else f"[{i}] ({chunk['metadata']['company']} {chunk['metadata']['year']} "
                    f"10-K, page {chunk['metadata']['page']})"
                )
                guarded.append(f"{tag}\n{wrap_document(clean_text)}")
            evidence = "\n\n".join(guarded)
            if redacted_total:
                print(f"  (guards: {redacted_total} injected line(s) redacted)", flush=True)

        answer_text = generate_answer(question, evidence, trace=trace)
    except Exception as exc:
        end_query(trace, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        raise

    end_query(trace)
    return {
        "route": tool,
        "search_query": search_query,
        "sql_debug": sql_debug,
        "graph_debug": graph_debug,
        "answer": answer_text,
        "trace": trace,
    }
