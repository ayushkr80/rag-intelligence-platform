"""Phase 2 extraction: raw PDFs -> per-page markdown JSON in data/processed.

pymupdf4llm reads each PDF page as markdown, preserving tables as pipe-table
text where it can. Every page record carries company/year/page metadata so any
chunk we later create inherits provenance — citations need this.
"""

import json
import sys
import time

import pymupdf4llm

from src import config

FILINGS = {
    "aapl-2024": ("Apple Inc.", 2024),
    "msft-2024": ("Microsoft Corporation", 2024),
    "nvda-2025": ("NVIDIA Corporation", 2025),
}


def main() -> None:
    wanted = sys.argv[1:] or None
    for pdf_path in sorted(config.RAW_DATA_DIR.glob("*.pdf")):
        if wanted and pdf_path.stem not in wanted:
            continue
        company, year = FILINGS[pdf_path.stem]

        print(f"extracting {pdf_path.name}...", flush=True)
        start = time.perf_counter()
        pages = pymupdf4llm.to_markdown(
            str(pdf_path), page_chunks=True, show_progress=True
        )
        elapsed = time.perf_counter() - start

        records = [
            {
                "company": company,
                "year": year,
                "page": page.get("metadata", {}).get("page", index),
                "text": page["text"],
            }
            for index, page in enumerate(pages)
        ]
        out_path = config.PROCESSED_DATA_DIR / f"{pdf_path.stem}.json"
        out_path.write_text(json.dumps(records, indent=1), encoding="utf-8")

        n_chars = sum(len(record["text"]) for record in records)
        print(
            f"{pdf_path.stem}: {len(records)} pages, {n_chars:,} chars, "
            f"{elapsed:.1f}s -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
