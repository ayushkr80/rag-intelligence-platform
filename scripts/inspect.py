"""Inspect extraction quality: show how key text/tables survived PDF->markdown.

Usage: python -m scripts.inspect <filing-stem> <needle> [width]
"""

import json
import sys

from src import config


def show(stem: str, needle: str, width: int = 600) -> None:
    path = config.PROCESSED_DATA_DIR / f"{stem}.json"
    pages = json.loads(path.read_text(encoding="utf-8"))
    for record in pages:
        idx = record["text"].find(needle)
        if idx != -1:
            start = max(0, idx - 150)
            snippet = record["text"][start : idx + width]
            print(f"=== {stem} page {record['page']} (match: {needle!r}) ===")
            print(snippet)
            return
    print(f"NOT FOUND: {needle!r} in {stem}")


if __name__ == "__main__":
    show(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600)
