"""Central configuration: loads environment variables and project paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.6-flash")


def active_api_key() -> str | None:
    """Return the API key for the configured provider, if present."""
    keys = {
        "openai": OPENAI_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
        "google": GOOGLE_API_KEY,
    }
    return keys.get(LLM_PROVIDER)


def ensure_dirs() -> None:
    """Create the data directories if they do not exist yet."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)
