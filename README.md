# Document Intelligence Platform

An enterprise-grade, evaluation-driven RAG system over SEC 10-K filings: hybrid retrieval,
agentic query routing (vector search / knowledge graph / SQL), permission-aware access,
and full observability.

**Live demo:** *(add your Hugging Face Space URL here after deploying — see Deployment)*

## What it does

Ask natural-language questions over hundreds of messy financial filings and get
cited, permission-checked answers — including multi-hop questions ("which supplier
does NVIDIA depend on?") and exact-number questions ("which company grew its cloud
business fastest?") that defeat plain vector search.

## Why

Naive "embed-and-retrieve" chatbots fail on real documents: tables turn to garbage,
multi-hop questions retrieve nothing useful, and nobody can verify the answers.
This project treats RAG as an engineered system: measured retrieval quality,
ablation-tested components, traced and cost-accounted generation.

## Architecture

```
                    ┌────────────┐
  10-K PDFs ──────▶ │ Ingestion  │──▶ chunks + tables + metadata
                    └────────────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
     vector index    knowledge graph      SQL (financials)
   (dense + BM25)     (networkx, 58n)      (SQLite, 62 rows)
          └────────────────┬──────────────────┘
                           ▼
   question ─▶ rewrite ─▶ ROUTER ─┬─▶ vector: hybrid + RRF ─▶ reranker ─┐
                                  ├─▶ sql: text-to-SQL (read-only) ─────┤
                                  └─▶ graph: 2-hop traversal ───────────┤
                                                                        ▼
                                              grounded LLM ──▶ cited answer
```

- **Hybrid retrieval**: from-scratch Okapi BM25 + dense embeddings, fused with
  reciprocal rank fusion, reranked by a local cross-encoder (`bge-reranker-base`)
- **Agentic routing**: an LLM classifier picks the cheapest sufficient tool per query
- **Guards**: permission filtering inside the retrieval layer (over-fetch → filter →
  rank), injection redaction, documents fenced as untrusted data
- **Resilience**: exponential backoff + jitter, model failover chains with circuit
  breakers, checkpointed index builds, query-embedding cache
- **Observability**: every query traced (model, tokens, latency, estimated cost)

Full build story: [docs/writeup.md](docs/writeup.md) · Resume bullets:
[docs/resume_bullets.md](docs/resume_bullets.md)

## Results

Golden set: 25 hand-verified questions across 7 categories (numeric, comparison,
multi-hop, explanatory, factoid, security, permission). Graders tiered from
deterministic fact-matching (free) to a rubric-based LLM judge.

**Retrieval ablation** (source hit@4, permission leaks = 0 across all modes):

| mode | source hit@4 |
|---|---|
| dense | 56% |
| hybrid (BM25 + dense, RRF) | 40% |
| hybrid + cross-encoder rerank | 52% |

**Agent end-to-end** (latest complete grading): route accuracy 22/24 (92%),
deterministic pass 17/24 (71%) — with eval-driven fixes (derivation clause,
BM25 stemming, router rules) verified and queued for re-grading. Judge
averages: 1.64/2 correctness, 1.52/2 faithfulness (calibrated against the
deterministic grader; documented where the judge is stricter than ground truth).

**Cost & latency** (from the trace log): ~$0.0002 per query; p50 ≈ 14s including
free-tier failover retries. Generation dominates spend — the case for routing.

**The evals caught 5 real bugs pre-ship** — a router regression, a graph-traversal
truncation, a guard design flaw, a SQL vocabulary mismatch, and golden-set
contamination. Run `scripts/run_evals.py --gate` — CI fails on any change that
drops hit-rate more than 2 points.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (then plain `python` uses the venv)
pip install -r requirements.txt
copy .env.example .env        # then add your API key
```

Always run modules with the venv interpreter: `.venv\Scripts\python -m scripts.chat`

## Run

```bash
.venv\Scripts\python -m scripts.chat          # interactive CLI (choose role)
.venv\Scripts\python -m uvicorn src.api:app --port 8000   # web UI on http://127.0.0.1:8000
.venv\Scripts\python -m scripts.run_evals --gate          # regression gate
.venv\Scripts\python -m scripts.trace_report              # cost/latency report
```

## Deployment

Ships with a `Dockerfile` (code + processed index; no raw data, no secrets).
Free-tier option — Hugging Face Spaces:

1. Create a Space (SDK: **Docker**, Public)
2. Upload `src/`, `static/`, `data/processed/`, `Dockerfile`, `requirements.txt`
3. Settings → Secrets → add `GOOGLE_API_KEY`
4. The Space builds and serves the UI with a public URL

Render/Railway work identically from the same Dockerfile. Note: the public
demo shares your API key's free-tier quota — consider a daily cap (the trace
log gives you the numbers to set one).

## Roadmap

- [x] Phase 0 — repo scaffold, config, environment
- [x] Phase 1 — naive RAG from scratch (embeddings, chunking, vector search)
- [x] Phase 2 — ingestion pipeline for messy PDFs (tables, metadata)
- [x] Phase 3 — hybrid retrieval + reranking + query rewriting
- [x] Phase 4 — agentic routing: vector / GraphRAG / SQL
- [x] Phase 5 — permission-aware retrieval, injection defense, citations
- [x] Phase 6 — eval pipeline (golden set, graders, LLM-as-judge, regression gate)
- [x] Phase 7 — observability (traces, latency, cost)
- [x] Phase 8 — FastAPI + UI + deployment (Dockerfile; Space URL above)
- [x] Phase 9 — writeup, resume packaging, demo prep

## Known limitations

- Reranker depends on torch, blocked intermittently by Windows application-control
  policy — the system degrades to hybrid ranking and flags it in evals
- SQL coverage: geographic segments extracted for Apple only; NVIDIA platform
  rows pending (tracked in eval failures Q3/Q23)
- 4 golden questions re-grade on the next eval run after the latest fixes
