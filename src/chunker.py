"""Chunker: splits a document into embeddable pieces.

Strategy for Phase 1: pack paragraphs into chunks under a size limit, carrying
one paragraph of overlap between consecutive chunks so ideas that straddle a
boundary live fully inside at least one chunk.
"""

MAX_CHARS = 800
OVERLAP_PARAGRAPHS = 1


def chunk_text(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap_paragraphs: int = OVERLAP_PARAGRAPHS,
) -> list[str]:
    """Split text into chunks of at most ~max_chars, each starting with the
    previous chunk's last paragraph as overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current and current_len + len(para) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paragraphs:]
            current_len = sum(len(p) + 2 for p in current)
        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks
