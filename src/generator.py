"""LLM generation: turns retrieved context + a question into a grounded answer."""

from google import genai

from src import config

_client: genai.Client | None = None

PROMPT_TEMPLATE = """You answer questions using ONLY the context below.

Context:
{context}

Question: {question}

Rules:
- Use only facts stated in the context; do not use outside knowledge.
- Cite the chunk numbers you used, like [0] or [1][2].
- If the context does not contain the answer, reply exactly: Not in the documents.
"""


def get_client() -> genai.Client:
    """Lazily create one shared Gemini client for the whole process."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.active_api_key())
    return _client


def generate_answer(question: str, context: str) -> str:
    """Answer the question grounded in the numbered context chunks."""
    response = get_client().models.generate_content(
        model=config.GEMINI_GENERATION_MODEL,
        contents=PROMPT_TEMPLATE.format(context=context, question=question),
    )
    return response.text
