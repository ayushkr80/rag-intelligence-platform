# Build Writeup — Document Intelligence Platform

## What I set out to prove

Everyone has a RAG demo. Almost nobody has a RAG system they can prove works.
The goal of this project was to build retrieval-augmented generation the way
an engineering team would: measured, secured, traced, and honest about its
own failures — over a corpus nobody hand-picks for cleanliness: real SEC 10-K
filings from Apple, Microsoft, and NVIDIA (355 pages, tables, headers, noise).

## What the system does

A question enters a query rewriter (follow-ups like "what about Microsoft?"
become standalone queries), then a router classifies it to the cheapest
sufficient tool:

- **vector** — hybrid dense + BM25 retrieval fused with reciprocal rank
  fusion, then a local cross-encoder reranker, for meaning-based questions
- **sql** — text-to-SQL over a 62-row financials table extracted from the
  filings, for exact numbers and cross-company comparisons
- **graph** — a 58-node knowledge graph (companies, executives, products,
  suppliers, competitors) built offline by LLM extraction and traversed with
  zero API calls at query time

Answers are generated grounded in tool evidence with page-level citations,
permission checks enforced inside the retrieval layer, and injection
redaction on everything retrieved. Every query is traced: model served,
tokens, latency, estimated cost (~$0.0002/query).

## The stories that mattered

**The quota saga taught resilience.** Free-tier limits (100 embeds/minute,
1000/day; 20 generations/day per model) forced the architecture I would have
built anyway in production: exponential backoff with jitter, checkpointed
index builds that resume after interruption, a model failover chain, and a
circuit breaker that skips drained models instantly.

**The evals overturned my own assumptions.** I believed hybrid retrieval was
strictly better. The ablation said dense beat hybrid on company-specific
hits (60% vs 44% hit@4) because BM25 crosses company boundaries. I believed
our graph could answer "who is NVIDIA's CEO?" — it couldn't, because a graph
of `led_by` edges never states titles, and the honest refusal was the system
working. Both findings are documented, not hidden.

**The evals caught five real bugs before they shipped**: a router regression
I introduced myself, a knowledge-graph truncation that evicted the matched
entity's own facts, a guard design that destroyed benign content along with
injected payloads, a SQL vocabulary mismatch, and two golden-set errors
contaminated by build-time assumptions.

**Observability made cost a design input.** After tracing, generation was
obviously the cost/latency dominator — which is the entire argument for
routing cheap questions to cheap tools.

## What's next

Deeper eval coverage (100+ questions), a nightly regression job, streaming
responses, and migration of the in-memory indexes to Postgres/pgvector.
