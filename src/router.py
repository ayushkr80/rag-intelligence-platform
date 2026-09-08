"""Router: classify a question to exactly one retrieval tool.

The smallest useful agent: one LLM call, one word out. Routing is cheap and
keeps each tool's prompt focused — far better than one mega-prompt that
retrieves everywhere and reasons nowhere.
"""

from src.config import FALLBACK_MODELS, TOOL_MODEL
from src.generator import get_client
from src.retry import call_with_failover

TOOLS = ("vector", "sql")

ROUTE_PROMPT = """Classify the question to exactly one retrieval tool.

- vector: meaning-based search over document text. Best for explanations, definitions, qualitative discussion, risk factors, "why" questions.
- sql: exact numbers from a structured financial table (revenue, net income, R&D spend, dividends, segment revenues by fiscal year). Best for "how much" questions, comparisons across companies or years, growth rates.

Examples:
"Which supplier is shared by both companies?" -> vector
"What was Microsoft's total revenue in fiscal 2024?" -> sql
"How much dividend per share did Apple pay?" -> sql
"Why did gross margin improve?" -> vector
"Which company grew its data center business fastest?" -> sql
"What risks does Apple disclose about supply chain?" -> vector

Question: {question}

Reply with exactly one word: vector or sql."""


def route(question: str) -> str:
    """Return 'vector' or 'sql' for the question."""
    response = call_with_failover(
        [TOOL_MODEL, *FALLBACK_MODELS],
        lambda model: get_client().models.generate_content(
            model=model,
            contents=ROUTE_PROMPT.format(question=question),
        ),
        label="route",
    )
    tool = response.text.strip().lower()
    return tool if tool in TOOLS else "vector"
