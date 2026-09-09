"""Security demo: permission-aware retrieval + prompt-injection defense.

Part A costs zero LLM calls — permission filtering is pure local policy
enforced inside retrieval. Part B runs the agent once against a live
injection payload smuggled in a vendor document.

Run parts: python -m scripts.security_demo A   (or B, or both)
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent import answer
from src.embeddings import embed_query_cached
from src.retrieval import load_corpus, retrieve

COMP_QUESTION = "What does Apple's Summary Compensation Table show?"
INJECTION_QUESTION = "Summarize the vendor assessment document."


def permission_demo(store, bm25) -> None:
    print("A. PERMISSION-AWARE RETRIEVAL (no LLM involved)\n")
    query_vector = embed_query_cached(COMP_QUESTION)
    for role in ("analyst", "employee"):
        results = retrieve(
            query_vector, COMP_QUESTION, store, bm25, "hybrid_rerank", role=role
        )
        print(f"  role={role}: {len(results)} results")
        for chunk in results[:3]:
            meta = chunk["metadata"]
            preview = chunk["text"][:60].replace("\n", " ")
            print(f"    [{meta['company']} p.{meta['page']} | {meta['level']}] {preview}...")
    print()


def injection_demo() -> None:
    print("B. PROMPT-INJECTION DEFENSE (live agent)")
    result = answer([], INJECTION_QUESTION, role="analyst")
    print(f"  route: {result['route']}")
    print(f"  ANSWER: {result['answer']}")


def main() -> None:
    parts = set(sys.argv[1:]) or {"A", "B"}
    store, bm25, chunks = load_corpus()
    if "A" in parts:
        permission_demo(store, bm25)
    if "B" in parts:
        injection_demo()


if __name__ == "__main__":
    main()
