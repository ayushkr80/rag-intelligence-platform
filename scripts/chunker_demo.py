"""See how a document becomes chunks — and how overlap keeps context across boundaries."""

from src.chunker import chunk_text
from src.sample_data import SAMPLE_FILING

chunks = chunk_text(SAMPLE_FILING, max_chars=300, overlap_paragraphs=1)
print(f"{len(chunks)} chunks\n")
for i, chunk in enumerate(chunks):
    print(f"--- chunk {i} ({len(chunk)} chars) ---")
    print(chunk)
    print()
