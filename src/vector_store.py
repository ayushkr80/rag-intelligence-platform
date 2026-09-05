"""In-memory vector store: chunks + embeddings, searched by cosine similarity.

Phase 1 keeps everything in RAM with numpy. Later phases swap this for
Postgres+pgvector / Qdrant — the interface (add, search) stays the same.
"""

import numpy as np


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self._vectors: np.ndarray | None = None

    def add(self, texts: list[str], vectors: list[list[float]]) -> None:
        """Store chunks alongside their embedding vectors."""
        batch = np.array(vectors)
        self._vectors = batch if self._vectors is None else np.vstack([self._vectors, batch])
        self.texts.extend(texts)

    def search(self, query_vector: list[float], top_k: int = 3) -> list[tuple[float, str]]:
        """Return the (score, text) of the top_k most similar chunks."""
        norms = self._vectors / np.linalg.norm(self._vectors, axis=1, keepdims=True)
        query = np.array(query_vector) / np.linalg.norm(query_vector)
        scores = norms @ query
        top = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.texts[i]) for i in top]
