"""BM25 (Okapi) keyword search, implemented from scratch.

Dense embeddings find meaning; BM25 finds exact terms — fund names, clause
numbers, "$245,122" — that embeddings average away. Pure local computation:
no API, no quota.
"""

import math
import re

K1 = 1.5
B = 0.75

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self.docs: list[list[str]] = []
        self.doc_freqs: dict[str, int] = {}
        self.doc_lens: list[int] = []

    def add(self, texts: list[str]) -> None:
        for text in texts:
            tokens = tokenize(text)
            self.docs.append(tokens)
            self.doc_lens.append(len(tokens))
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def search(self, query: str, top_k: int = 4) -> list[tuple[float, int]]:
        """Return (score, doc_index) for the top_k documents."""
        query_terms = tokenize(query)
        n_docs = len(self.docs)
        avg_len = sum(self.doc_lens) / n_docs
        scores = [0.0] * n_docs

        for term in query_terms:
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for i, doc in enumerate(self.docs):
                tf = doc.count(term)
                if tf == 0:
                    continue
                norm = K1 * (1 - B + B * self.doc_lens[i] / avg_len)
                scores[i] += idf * tf * (K1 + 1) / (tf + norm)

        ranked = sorted(range(n_docs), key=lambda i: scores[i], reverse=True)
        return [(scores[i], i) for i in ranked[:top_k] if scores[i] > 0]
