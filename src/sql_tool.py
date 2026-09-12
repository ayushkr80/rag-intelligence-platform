"""Text-to-SQL tool: exact numbers from the structured financials table.

Security first: the LLM may only produce a single read-only SELECT. Anything
else — multiple statements, data modification — is blocked before execution.
"""

import sqlite3

from google.genai import types

from src import config
from src.config import FALLBACK_MODELS, TOOL_MODEL
from src.generator import get_client
from src.retry import call_with_failover

DB_PATH = config.PROCESSED_DATA_DIR / "financials.db"

FORBIDDEN_WORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "pragma", "vacuum", "replace",
)

SQL_PROMPT = """Write ONE SQLite SELECT query that answers the question.

Table: financials(company TEXT, metric TEXT, segment TEXT, fiscal_year INTEGER, value REAL, unit TEXT, source_page INTEGER)
- metric is one of: total_revenue, net_income, rnd_expense, dividends_per_share, segment_revenue
- segment is one of: services, intelligent_cloud, productivity_and_business, data_center, gaming, professional_visualization, automotive, compute_networking, graphics, other — NULL for company-level metrics
- NVIDIA's Data Center platform revenue is stored under segment='compute_networking'; the value 'data_center' DOES NOT EXIST in this table — never filter on it. Microsoft's cloud business is 'intelligent_cloud'; Apple's is 'services'
- fiscal years differ per company (Apple/Microsoft end 2024, NVIDIA ends 2025) — compute growth within each company's own years
- unit: 'MUSD' (millions of USD) or 'USD_per_share'
- companies: 'Apple Inc.', 'Microsoft Corporation', 'NVIDIA Corporation'

Rules:
- Exactly one statement, starting with SELECT, no semicolon, no comments.
- Read-only: never modify data.
- Prefer SIMPLE SELECTs returning raw rows over complex arithmetic — the caller computes growth rates and comparisons from the rows. Avoid subqueries on MAX/MIN fiscal_year.

Question: {question}

Reply with ONLY the SQL."""


def run_sql(question: str) -> dict:
    """Generate, validate, and execute a read-only query. Returns rows or error."""
    response = call_with_failover(
        [TOOL_MODEL, *FALLBACK_MODELS],
        lambda model: get_client().models.generate_content(
            model=model,
            contents=SQL_PROMPT.format(question=question),
            config=types.GenerateContentConfig(temperature=0),
        ),
        label="sql",
    )
    sql = response.text.strip().strip("`").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()

    lowered = sql.lower()
    if not lowered.startswith("select") or any(word in lowered for word in FORBIDDEN_WORDS):
        return {"sql": sql, "error": "blocked: only single read-only SELECT allowed"}

    connection = sqlite3.connect(DB_PATH)
    try:
        rows = connection.execute(sql).fetchall()
    finally:
        connection.close()
    return {"sql": sql, "rows": rows}
