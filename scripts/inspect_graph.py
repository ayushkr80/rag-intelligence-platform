"""Print knowledge-graph edges: subject | relation | object | source page."""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.graph_tool import load_graph

graph = load_graph()
print(f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n")
for u, v, data in graph.edges(data=True):
    print(f"{u} | {data.get('relation')} | {v} | p.{data.get('source_page')}")
