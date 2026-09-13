# Resume Bullets

Ready-to-adapt bullets — replace the numbers with the latest eval run.

## AI / ML Engineer resume

- Built an evaluation-driven enterprise RAG platform over 355 pages of SEC
  10-K filings: hybrid retrieval (dense + BM25 + reciprocal rank fusion),
  cross-encoder reranking, and agentic query routing across vector search,
  text-to-SQL, and knowledge-graph tools — improving measured answer
  accuracy from a 48% naive baseline to 84% on a 25-question golden set.
- Designed an automated eval pipeline (golden set, deterministic graders,
  calibrated LLM-as-judge, regression gate) that caught 5 real defects
  pre-ship, including a router regression and a graph-traversal truncation
  bug; CI-style gate fails any change dropping retrieval hit-rate >2 points.
- Implemented production resilience for LLM APIs: exponential backoff with
  jitter, checkpointed index builds resumable across daily quota resets,
  model failover chains with circuit breakers — zero data loss across
  interrupted 1,900-chunk embedding runs.
- Secured the retrieval layer with role-based access enforced at query time
  (over-fetch/filter/rank so restricted documents never reach the model) and
  layered prompt-injection defenses (detection, line-level redaction,
  fenced untrusted context) proven against live payloads.
- Deployed as a FastAPI service with a traced chat UI; every query records
  model, tokens, latency (p50/p95), and estimated cost (~$0.0002/query),
  driving a cost-aware routing design where generation dominates spend.

## Interview soundbites (60 seconds each)

- **Debugging RAG:** "When an answer is wrong but the document exists, I
  classify the failing stage first — retrieval coverage vs ranking vs
  generation. Our evals found all three species: BM25 pulling cross-company
  noise, a reranker evicting the right chunk, and a prompt that refused to
  compute derivable answers."
- **When GraphRAG beats vectors:** "Multi-hop relational questions — shared
  suppliers, who leads whom. We precompute the graph offline so query time
  is free; vector search can't follow two hops."
- **Trusting an LLM judge:** "A judge is a measurement instrument. We
  calibrate it against deterministic fact-matching and documented where it's
  stricter — penalizing true facts beyond the golden answer."
