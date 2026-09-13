"""FastAPI server exposing the agent as an HTTP API with a bundled chat UI.

Endpoints:
- GET  /health — liveness
- GET  /       — the chat UI (static/index.html)
- POST /ask    — {question, role, history} -> {answer, route, latency, cost}

Security: client-supplied roles are validated against the permission policy;
unknown roles silently downgrade to least privilege (employee is still a
known role, so anything unrecognized becomes the most-restricted default).
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.agent import answer
from src.permissions import ROLES

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.retrieval import load_corpus

    load_corpus()
    yield


app = FastAPI(title="Document Intelligence Platform", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    role: str = "employee"
    history: list[tuple[str, str]] = []


class AskResponse(BaseModel):
    answer: str
    route: str
    duration_ms: int | None = None
    cost_usd: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    role = request.role if request.role in ROLES else next(iter(ROLES))
    result = answer(request.history, request.question.strip(), role=role)
    trace = result.get("trace", {})
    return AskResponse(
        answer=result["answer"],
        route=result["route"],
        duration_ms=trace.get("duration_ms"),
        cost_usd=trace.get("cost_usd"),
    )
