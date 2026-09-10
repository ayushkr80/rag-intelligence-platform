"""GraphRAG tool: multi-hop relational lookup over the prebuilt knowledge graph.

The graph is precomputed offline (extraction cost paid once); querying it is
pure local traversal — zero API calls per question. This is the GraphRAG
trade: pay at build time, answer relational hops cheaply at query time.
"""

import json
from functools import lru_cache

import networkx as nx

from src import config

GRAPH_PATH = config.PROCESSED_DATA_DIR / "graph.json"
MAX_HOPS = 2
MAX_TRIPLES = 25


@lru_cache(maxsize=1)
def load_graph() -> nx.Graph:
    """Load the persisted knowledge graph once per process."""
    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    return nx.node_link_graph(data)


def find_entities(question: str, graph: nx.Graph) -> list[str]:
    """Match question text against known entities without any API call."""
    question_lower = question.lower()
    matches = []
    for node in graph.nodes:
        node_lower = str(node).lower()
        if len(node_lower) < 4:
            continue
        if node_lower in question_lower:
            matches.append(node)
            continue
        first_word = node_lower.split()[0] if node_lower.split() else ""
        if len(first_word) >= 4 and first_word in question_lower:
            matches.append(node)
    return matches


def neighborhood_triples(
    graph: nx.Graph, entities: list[str], max_hops: int = MAX_HOPS
) -> list[tuple[str, str, str, int]]:
    """Collect edges within max_hops of the matched entities.

    Direct edges (hop 1) always rank before hop-2 context, then by page —
    so one noisy neighbor can never evict the matched entity's own facts.
    """
    scored: list[tuple[int, int, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    current = set(entities)
    for hop in range(1, max_hops + 1):
        next_nodes: set[str] = set()
        for node in current:
            for neighbor, edge_data in graph[node].items():
                relation = edge_data.get("relation", "related_to")
                key = (node, relation, neighbor)
                if key in seen:
                    continue
                seen.add(key)
                scored.append(
                    (hop, edge_data.get("source_page", 0), node, relation, neighbor)
                )
                next_nodes.add(neighbor)
        current = next_nodes
    scored.sort(key=lambda item: (item[0], item[1]))
    return [
        (subject, relation, obj, page)
        for hop, page, subject, relation, obj in scored[:MAX_TRIPLES]
    ]


def query(question: str) -> dict:
    """Return matched entities and their neighborhood triples."""
    graph = load_graph()
    entities = find_entities(question, graph)
    triples = neighborhood_triples(graph, entities) if entities else []
    return {"entities": entities, "triples": triples}
