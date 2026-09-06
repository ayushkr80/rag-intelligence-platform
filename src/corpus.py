"""Corpus assembly: page records -> metadata-rich chunks ready to embed.

Chunks are cut within a single page so provenance stays exact: every chunk
knows its company, fiscal year, and source page for citations.
"""

import json

from src.chunker import chunk_text
from src import config

MIN_CHUNK_CHARS = 60
MAX_CHARS = 800
OVERLAP_PARAGRAPHS = 1


def load_pages(stem: str) -> list[dict]:
    path = config.PROCESSED_DATA_DIR / f"{stem}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_filing(pages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for record in pages:
        for piece in chunk_text(record["text"], MAX_CHARS, OVERLAP_PARAGRAPHS):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                {
                    "company": record["company"],
                    "year": record["year"],
                    "page": record["page"],
                    "text": piece,
                }
            )
    return chunks
