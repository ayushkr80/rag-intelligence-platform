# Document Intelligence Platform

An enterprise-grade, evaluation-driven RAG system over SEC 10-K filings: hybrid retrieval,
agentic query routing (vector search / knowledge graph / SQL), permission-aware access,
and full observability.

## What it does

Ask natural-language questions over hundreds of messy financial filings and get
cited, permission-checked answers — including multi-hop questions ("which supplier
do both companies share?") and exact-number questions ("segment revenue for 2024")
that defeat plain vector search.

## Why

Naive "embed-and-retrieve" chatbots fail on real documents: tables turn to garbage,
multi-hop questions retrieve nothing useful, and nobody can verify the answers.
This project treats RAG as an engineered system: measured retrieval quality,
ablation-tested components, traced and cost-accounted generation.

## Architecture

```
                 ┌────────────┐
  10-K PDFs ───▶ │ Ingestion  │──▶ chunks + tables + metadata
                 └────────────┘
                        │
        ┌───────────────┼─────────────────┐
        ▼               ▼                 ▼
   vector index      graph (Neo4j)     SQL (tables)
        └───────┬───────┴─────────────────┘
                ▼
        query router (agent) ──▶ reranker ──▶ LLM ──▶ cited answer
```

*(Detailed component docs land in `docs/` as each phase ships.)*

## Results

*(Filled from the eval pipeline in Phase 6 — ablation table of each component's
contribution to accuracy, faithfulness, and context precision.)*

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (then plain `python` uses the venv)
pip install -r requirements.txt
copy .env.example .env        # then add your API key
```

Always run modules with the venv interpreter: `.venv\Scripts\python -m scripts.chat`

## Roadmap

- [x] Phase 0 — repo scaffold, config, environment
- [x] Phase 1 — naive RAG from scratch (embeddings, chunking, vector search)
- [x] Phase 2 — ingestion pipeline for messy PDFs (tables, metadata)
- [x] Phase 3 — hybrid retrieval + reranking + query rewriting
- [ ] Phase 4 — agentic routing: vector / GraphRAG / SQL
- [ ] Phase 5 — permission-aware retrieval, injection defense, citations
- [ ] Phase 6 — eval pipeline (golden set, Ragas, LLM-as-judge, regression gate)
- [ ] Phase 7 — observability (traces, latency, cost)
- [ ] Phase 8 — FastAPI + UI + deployment
- [ ] Phase 9 — writeup, resume packaging, demo prep
