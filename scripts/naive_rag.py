"""The complete naive RAG: chunk -> embed -> store -> retrieve -> grounded answer.

Every question goes through the full pipeline. The third question is designed
to fail — it tests whether the system admits the answer is not in the documents
instead of hallucinating.
"""

from src.chunker import chunk_text
from src.embeddings import embed_query, embed_texts
from src.generator import generate_answer
from src.sample_data import SAMPLE_FILING
from src.vector_store import InMemoryVectorStore

QUESTIONS = [
    "How did cloud services growth compare to hardware revenue?",
    "What percentage of key components come from two suppliers?",
    "How much dividend did Acme pay per share in 2024?",
    "Why did gross margin improve in 2024?",
    "Who is Acme's auditor?"
]

chunks = chunk_text(SAMPLE_FILING, max_chars=300, overlap_paragraphs=1)
vectors = embed_texts(chunks)

store = InMemoryVectorStore()
store.add(chunks, vectors)

for question in QUESTIONS:
    print("=" * 70)
    print("QUESTION:", question)

    results = store.search(embed_query(question), top_k=3)
    for score, text, _ in results:
        preview = text[:70].replace("\n", " ")
        print(f"  {score:.3f}  {preview}...")

    context = "\n\n".join(
        f"[{i}] {text}" for i, (_, text, _) in enumerate(results)
    )
    print("\nANSWER:", generate_answer(question, context))
    print()
