"""Build the searchable index: chunk all filings, embed, persist to disk.

Checkpointed: vectors are saved after every batch, so an interrupted run
resumes where it stopped instead of re-embedding from scratch.
"""

import json
import time

import numpy as np

from src import config
from src.corpus import chunk_filing, load_pages
from src.embeddings import embed_texts

BATCH_SIZE = 24
SLEEP_BETWEEN_BATCHES = 0.5
VECTOR_PATH = config.PROCESSED_DATA_DIR / "index_vectors.npy"
CHUNK_PATH = config.PROCESSED_DATA_DIR / "index_chunks.json"
CHECKPOINT_PATH = config.PROCESSED_DATA_DIR / "index_progress.json"


def main() -> None:
    start = time.perf_counter()
    chunks: list[dict] = []
    for stem in ("aapl-2024", "msft-2024", "nvda-2025"):
        filing_chunks = chunk_filing(load_pages(stem))
        print(f"{stem}: {len(filing_chunks)} chunks", flush=True)
        chunks.extend(filing_chunks)

    vectors: list[list[float]] = []
    done = 0
    if CHECKPOINT_PATH.exists():
        state = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        done = state["done"]
        vectors = np.load(VECTOR_PATH).tolist()
        print(f"resuming from checkpoint: {done}/{len(chunks)} chunks embedded", flush=True)

    print(f"total: {len(chunks)} chunks — embedding...", flush=True)
    (CHUNK_PATH).write_text(json.dumps(chunks, indent=1), encoding="utf-8")
    while done < len(chunks):
        batch = [chunk["text"] for chunk in chunks[done : done + BATCH_SIZE]]
        vectors.extend(embed_texts(batch))
        done += len(batch)
        np.save(VECTOR_PATH, np.array(vectors, dtype=np.float32))
        CHECKPOINT_PATH.write_text(json.dumps({"done": done}), encoding="utf-8")
        print(f"  embedded {done}/{len(chunks)}", flush=True)
        time.sleep(SLEEP_BETWEEN_BATCHES)

    CHECKPOINT_PATH.unlink(missing_ok=True)
    print(f"done in {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
