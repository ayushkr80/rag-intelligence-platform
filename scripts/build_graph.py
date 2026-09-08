"""One-time build: extract knowledge-graph triples from filings into networkx.

Anchor pages (executives, human capital, supplier/risk discussion, business
overview) feed one LLM extraction per company. The graph persists as JSON so
query-time GraphRAG is pure local traversal.
"""

import json
import re
import time

import networkx as nx

from src import config
from src.config import FALLBACK_MODELS, TOOL_MODEL
from src.generator import get_client
from src.retry import call_with_failover

GRAPH_PATH = config.PROCESSED_DATA_DIR / "graph.json"

COMPANIES = {
    "aapl-2024": "Apple Inc.",
    "msft-2024": "Microsoft Corporation",
    "nvda-2025": "NVIDIA Corporation",
}

MAX_PAGES = 8
PAGE_CHARS = 3500

EXTRACTION_PROMPT = """You extract knowledge-graph triples from 10-K page excerpts.

Company: {company}

Page excerpts (each labeled with its PDF page number):
{pages}

Extract factual triples as STRICT JSON — a list, no markdown fences, no commentary:
[{{"subject": "...", "subject_type": "company|person|product|segment|supplier|partner|location|technology|other", "relation": "manufactures|offers|led_by|headquartered_in|depends_on_supplier|supplies_to|competes_with|operates_segment|invests_in|other", "object": "...", "object_type": "...", "source_page": 12}}]

Rules:
- Only facts actually present in the excerpts; one fact per triple; cite the page.
- Include: key products and segments (offers/operates_segment), executive officers (led_by), named suppliers and outsourcing partners (depends_on_supplier), headquarters (headquartered_in), major partners.
- Normalize entity names consistently (e.g. always "NVIDIA Corporation", never "Nvidia" and "NVIDIA Corporation").
- Return [] if nothing extractable."""


def find_pages(pages: list[dict]) -> list[dict]:
    selected = []
    for record in pages:
        text = record["text"]
        wanted = (
            "Executive Officers" in text
            or "Human Capital" in text
            or "Outsourcing" in text
            or ("supplier" in text.lower() and "depend" in text.lower())
            or record["page"] <= 4
        )
        if wanted:
            selected.append(record)
        if len(selected) >= MAX_PAGES:
            break
    return selected


def extract_triples(company: str, selected: list[dict]) -> list[dict]:
    pages_text = "\n\n".join(
        f"--- page {record['page']} ---\n{record['text'][:PAGE_CHARS]}"
        for record in selected
    )
    prompt = EXTRACTION_PROMPT.format(company=company, pages=pages_text)

    for attempt in range(2):
        response = call_with_failover(
            [TOOL_MODEL, *FALLBACK_MODELS],
            lambda model: get_client().models.generate_content(
                model=model, contents=prompt
            ),
            label="graph-extract",
        )
        text = response.text.strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            triples = json.loads(text)
            if isinstance(triples, list):
                return triples
        except json.JSONDecodeError:
            if attempt == 0:
                prompt += "\n\nREMINDER: reply with ONLY valid JSON."
                time.sleep(2)
    return []


def main() -> None:
    graph = nx.Graph()

    for stem, company in COMPANIES.items():
        pages = json.loads(
            (config.PROCESSED_DATA_DIR / f"{stem}.json").read_text(encoding="utf-8")
        )
        selected = find_pages(pages)
        print(f"{company}: {len(selected)} anchor pages", flush=True)

        graph.add_node(company, type="company")
        triples = extract_triples(company, selected)
        added = 0
        for triple in triples:
            try:
                subject = str(triple["subject"]).strip()
                obj = str(triple["object"]).strip()
                relation = str(triple["relation"]).strip()
                page = int(triple["source_page"])
                graph.add_edge(subject, obj, relation=relation, source_page=page)
                graph.nodes[subject].setdefault("type", triple.get("subject_type", "other"))
                graph.nodes[obj].setdefault("type", triple.get("object_type", "other"))
                added += 1
            except (KeyError, TypeError, ValueError) as exc:
                print(f"  skipped bad triple {triple}: {exc}", flush=True)
        print(f"  {added} triples added", flush=True)
        time.sleep(2)

    data = nx.node_link_data(graph)
    GRAPH_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(
        f"graph saved: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges -> {GRAPH_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
