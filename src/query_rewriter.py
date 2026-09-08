"""Query rewriting: turns follow-up questions into standalone search queries.

A retriever only sees the query — "What about Microsoft?" embeds to near-nothing.
Rewriting resolves pronouns and references against the conversation history so
every retrieval gets a self-contained query. First turn passes through unchanged
(no API call, no cost).
"""

from src.generator import get_client
from src.retry import call_with_backoff
from src import config

REWRITE_PROMPT = """Given the conversation history and the latest user question, rewrite the latest question as a standalone search query that can be understood without the history.

Rules:
- Resolve pronouns and references (e.g. "What about 2022?" must name the topic it refers to).
- Keep the question's language and key entities.
- If the question is already standalone, return it unchanged.
- Reply with ONLY the rewritten question, nothing else.

History:
{history}

Latest question: {question}"""

MAX_HISTORY_TURNS = 4
MAX_ANSWER_CHARS = 300


def rewrite_query(history: list[tuple[str, str]], question: str) -> str:
    """Return a standalone version of the latest question."""
    if not history:
        return question
    history_text = "\n".join(
        f"User: {asked}\nAssistant: {answered[:MAX_ANSWER_CHARS]}"
        for asked, answered in history[-MAX_HISTORY_TURNS:]
    )
    response = call_with_backoff(
        lambda: get_client().models.generate_content(
            model=config.GEMINI_GENERATION_MODEL,
            contents=REWRITE_PROMPT.format(history=history_text, question=question),
        ),
        label="rewrite",
    )
    return response.text.strip().strip('"')
