"""Agent orchestrator: rewrite -> route -> tool -> grounded answer.

The full Phase 4 pipeline. Every question gets the cheapest sufficient tool;
each tool returns evidence in the same format, so generation stays uniform.
"""

import time

from src import config
from src.embeddings import embed_query_cached
from src.generator import generate_answer
from src.query_rewriter import rewrite_query
from src.retrieval import load_corpus, retrieve
from src.router import route
from src.sql_tool import run_sql

_store = _bm25 = _chunks = None


def _corpus():
    global _store, _bm25, _chunks
    if _store is None:
        _store, _bm25, _chunks = load_corpus()
    return _store, _bm25, _chunks


def answer(history: list[tuple[str, str]], question: str) -> dict:
    """Run the full pipeline; returns route, search query, and the answer."""
    search_query = rewrite_query(history, question)
    tool = route(search_query)
    time.sleep(1.0)
    sql_debug = None

    if tool == "sql":
        result = run_sql(search_query)
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

    if tool != "sql":
        store, bm25, _ = _corpus()
        results = retrieve(
            embed_query_cached(search_query), search_query, store, bm25, config.RAG_MODE
        )
        evidence = "\n\n".join(
            f"[{i}] ({chunk['metadata']['company']} {chunk['metadata']['year']} 10-K, "
            f"page {chunk['metadata']['page']})\n{chunk['text']}"
            for i, chunk in enumerate(results)
        )

    return {
        "route": tool,
        "search_query": search_query,
        "sql_debug": sql_debug,
        "answer": generate_answer(question, evidence),
    }
