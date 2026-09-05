"""See embeddings working: similar meanings score high, unrelated score low."""

import numpy as np

from src.embeddings import embed_texts

sentences = [
    "Revenue grew 20% year over year",
    "Sales increased by a fifth compared to last year",
    "The CEO resigned amid a scandal",
    "The board fired its chief executive after public pressure",
]

vectors = np.array(embed_texts(sentences))
norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
similarity = norms @ norms.T

for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i < j:
            print(f"{similarity[i][j]:.3f}  '{sentences[i]}'  vs  '{sentences[j]}'")
