"""One-time build: extract key financials from filing pages into SQLite.

Page finding uses exact text anchors (structural anchors beat semantic search
for known artifacts, and cost zero API calls). The LLM then extracts strict
JSON rows; parsing is validated and retried once — structured output always
needs defense.
"""

import json
import re
import sqlite3
import time

from src import config
from src.config import TOOL_MODEL
from src.generator import get_client
from src.retry import call_with_backoff

DB_PATH = config.PROCESSED_DATA_DIR / "financials.db"

COMPANIES = {
    "aapl-2024": "Apple Inc.",
    "msft-2024": "Microsoft Corporation",
    "nvda-2025": "NVIDIA Corporation",
}

ANCHORS = (
    "STATEMENTS OF OPERATIONS",
    "INCOME STATEMENTS",
    "STATEMENTS OF INCOME",
    "Products and Services Performance",
    "Segment Operating Performance",
    "Reportable Segments",
    "Revenue, classified by significant product",
    "Capital Return Program",
    "Data Center revenue",
    "Compute & Networking",
)

MAX_PAGES = 10
PAGE_CHARS = 3500

EXTRACTION_PROMPT = """You extract structured financial data from 10-K page excerpts.

Company: {company}

Page excerpts (each labeled with its PDF page number):
{pages}

Extract data rows as STRICT JSON — a list of objects, no markdown fences, no commentary:
[{{"metric": "...", "segment": null, "fiscal_year": 2024, "value": 123.45, "unit": "MUSD", "source_page": 55}}]

Rules:
- metric is one of: total_revenue, net_income, rnd_expense, dividends_per_share, segment_revenue
- dividends_per_share: extract every per-share dividend figure you find (Capital Return Program sections, equity statements, dividend tables) for every fiscal year shown
- segment (only for segment_revenue): services, intelligent_cloud, productivity_and_business, data_center, gaming, professional_visualization, automotive, compute_networking, graphics, other; null otherwise
- NVIDIA's market platforms are data_center, gaming, professional_visualization, automotive; its reportable segments are compute_networking and graphics — use whichever the excerpt actually reports
- value is a number: millions of USD (unit "MUSD"), or USD per share (unit "USD_per_share")
- include every fiscal year shown for each metric
- only extract numbers actually present in the excerpts; set source_page to that page's number
- return [] if nothing extractable"""


def find_pages(pages: list[dict]) -> list[dict]:
    selected = []
    for record in pages:
        if any(anchor in record["text"] for anchor in ANCHORS):
            selected.append(record)
        if len(selected) >= MAX_PAGES:
            break
    return selected


def extract_rows(company: str, selected: list[dict]) -> list[dict]:
    pages_text = "\n\n".join(
        f"--- page {record['page']} ---\n{record['text'][:PAGE_CHARS]}"
        for record in selected
    )
    prompt = EXTRACTION_PROMPT.format(company=company, pages=pages_text)

    for attempt in range(2):
        response = call_with_backoff(
            lambda: get_client().models.generate_content(
                model=TOOL_MODEL, contents=prompt
            ),
            label="extract",
        )
        text = response.text.strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            rows = json.loads(text)
            if isinstance(rows, list):
                return rows
        except json.JSONDecodeError:
            if attempt == 0:
                prompt += "\n\nREMINDER: reply with ONLY valid JSON."
                time.sleep(2)
    return []


def main() -> None:
    connection = sqlite3.connect(DB_PATH)
    connection.execute("DROP TABLE IF EXISTS financials")
    connection.execute(
        """CREATE TABLE financials(
            company TEXT, metric TEXT, segment TEXT,
            fiscal_year INTEGER, value REAL, unit TEXT, source_page INTEGER)"""
    )

    total = 0
    for stem, company in COMPANIES.items():
        pages = json.loads(
            (config.PROCESSED_DATA_DIR / f"{stem}.json").read_text(encoding="utf-8")
        )
        selected = find_pages(pages)
        print(f"{company}: {len(selected)} anchor pages", flush=True)

        rows = extract_rows(company, selected)
        for row in rows:
            try:
                connection.execute(
                    "INSERT INTO financials VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        company,
                        row["metric"],
                        row.get("segment"),
                        int(row["fiscal_year"]),
                        float(row["value"]),
                        row["unit"],
                        int(row["source_page"]),
                    ),
                )
                total += 1
            except (KeyError, TypeError, ValueError) as exc:
                print(f"  skipped bad row {row}: {exc}", flush=True)
        print(f"  {len(rows)} rows extracted", flush=True)
        connection.commit()
        time.sleep(2)

    connection.close()
    print(f"done: {total} rows -> {DB_PATH}", flush=True)


if __name__ == "__main__":
    main()
