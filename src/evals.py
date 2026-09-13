"""Graders for the eval pipeline.

Three grader types, cheapest first:
1. deterministic answer matching — substring facts, refusal checks (free)
2. retrieval source hits — did expected sources surface in top-k (free)
3. LLM-as-judge — rubric scoring on failover models (costs quota)

Judges are measurement instruments, not ground truth: the answer_matches
check provides the deterministic calibration signal to compare against.
"""

import json
import re

from google.genai import types

from src.config import FALLBACK_MODELS, TOOL_MODEL
from src.generator import get_client
from src.retry import call_with_failover

JUDGE_PROMPT = """You grade RAG answers. Compare the answer to the ground truth and score it.

Question: {question}
Ground truth facts that a correct answer should contain: {facts}
{refusal_note}
Answer to grade:
{answer}

Score:
- "correctness": 0 (wrong or missing the facts), 1 (partially correct), 2 (contains the key facts or, when refusal is expected, correctly declines)
- "faithfulness": 0 (states things not supported by the ground truth context), 1 (mostly supported), 2 (fully supported)

Reply with STRICT JSON only: {{"correctness": 0-2, "faithfulness": 0-2, "reasoning": "<one sentence>"}}"""


def normalize(text: str) -> str:
    """Lowercase and strip commas/$ so '96,169' matches '96169' and '$96,169'."""
    return re.sub(r"[,$]", "", text.lower())


def answer_matches(answer: str, expected_facts: list[str], expect_refusal: bool) -> bool:
    """Deterministic grade: refusals must decline, facts must appear."""
    normalized = normalize(answer)
    if expect_refusal:
        return "not in the documents" in normalized
    return all(normalize(fact) in normalized for fact in expected_facts)


def source_hit(results: list[dict], source_hints: list[dict]) -> bool:
    """True if any retrieved chunk matches a hint (company required, page if given)."""
    for hint in source_hints:
        for chunk in results:
            meta = chunk["metadata"]
            if meta.get("company") != hint["company"]:
                continue
            if "page" not in hint or meta.get("page") == hint["page"]:
                return True
    return False


def judge_answer(question: str, answer: str, expected_facts: list[str], expect_refusal: bool) -> dict:
    """Rubric-based LLM judge. Returns correctness/faithfulness scores."""
    refusal_note = (
        "A correct answer here is a clear refusal (the information is not in the documents)."
        if expect_refusal
        else "A correct answer must contain the ground truth facts."
    )
    response = call_with_failover(
        [TOOL_MODEL, *FALLBACK_MODELS],
        lambda model: get_client().models.generate_content(
            model=model,
            contents=JUDGE_PROMPT.format(
                question=question,
                facts=expected_facts or "[]",
                refusal_note=refusal_note,
                answer=answer,
            ),
            config=types.GenerateContentConfig(temperature=0),
        ),
        label="judge",
        trace=None,
    )
    text = re.sub(r"^```(json)?|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
    try:
        scores = json.loads(text)
        return {
            "correctness": int(scores.get("correctness", 0)),
            "faithfulness": int(scores.get("faithfulness", 0)),
            "reasoning": str(scores.get("reasoning", "")),
        }
    except (ValueError, AttributeError):
        return {"correctness": -1, "faithfulness": -1, "reasoning": f"unparseable: {text[:120]}"}
