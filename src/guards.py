"""Guards for untrusted retrieved text: injection detection + containment.

Retrieved documents are attacker-controlled input. Two layers:
1. detect_injection — heuristic scan for instruction-override patterns,
   so flagged chunks can be withheld from the prompt entirely.
2. wrap_document — surviving text is fenced as data, so even undetected
   injections compete with an explicit system rule to never obey them.
"""

import re

INJECTION_PATTERNS = (
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above)",
    r"disregard\s+(all\s+|any\s+)?(previous|prior|instructions)",
    r"reveal\s+(your\s+|the\s+)?(system|instructions|prompt)",
    r"system\s+prompt",
    r"you\s+are\s+now",
    r"new\s+instructions?:",
    r"admin(istrator)?\s+mode",
)


def detect_injection(text: str) -> bool:
    """True if the text matches any instruction-override pattern."""
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def wrap_document(text: str) -> str:
    """Fence text as untrusted data for the generation prompt."""
    return f"<document>\n{text}\n</document>"
