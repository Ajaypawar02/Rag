"""FastAPI application: routers, OpenAPI docs and startup.

Run locally:  uvicorn app.main:app --port 8080 --reload
Swagger UI:   http://localhost:8080/docs     (ReDoc: /redoc, schema: /openapi.json)
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes import chat, decisions, documents, memories, metrics, sessions
from app.api.schemas import HealthOut

logger = logging.getLogger(__name__)

DESCRIPTION = """
Conversational RAG over technical documentation, orchestrated by a LangGraph multi-agent pipeline.

**Typical flow**
1. `POST /documents/upload` - add PDF / Markdown / HTML docs (sample docs are pre-seeded).
2. `POST /sessions` with header `X-User-Id` - returns a `session_id`.
3. `POST /sessions/{session_id}/chat` (or `/chat/stream` for SSE) - ask questions; follow-ups use the session history.
4. `GET /sessions/{session_id}/decisions` - see what each agent decided and why.

Sessions are isolated per user (`X-User-Id`); long-term memories are shared across a user's sessions.
"""

TAGS = [
    {"name": "documents", "description": "Upload and manage the documentation knowledge base."},
    {"name": "sessions", "description": "Conversation sessions (one LangGraph thread each)."},
    {"name": "chat", "description": "Ask questions within a session."},
    {"name": "explainability", "description": "Agent decision traces."},
    {"name": "memory", "description": "Cross-session user memory."},
    {"name": "metrics", "description": "Performance and memory metrics."},
    {"name": "health", "description": "Service health."},
]


def warm_up() -> None:
    """Create app tables and load models/connections once, so the first request is not slow."""
    from app.db.session import init_db
    from app.graph.graph import get_graph
    from app.memory.keyword_index import get_keyword_index
    from app.observability.tracing import configure_langsmith

    configure_langsmith()
    init_db()
    get_graph()  # Postgres pool, checkpointer + store setup, embedding model
    get_keyword_index()  # Chroma connection + BM25 index


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        warm_up()
    except Exception:  # keep serving /health so the problem is visible instead of crash-looping
        logger.exception("Startup warm-up failed; check PostgreSQL and Chroma")
    yield


app = FastAPI(
    title="Conversational RAG API",
    version="1.0.0",
    description=DESCRIPTION,
    openapi_tags=TAGS,
    lifespan=lifespan,
)

for module in (documents, sessions, chat, decisions, memories, metrics):
    app.include_router(module.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthOut, tags=["health"], summary="Check PostgreSQL and Chroma")
def health() -> HealthOut:
    from sqlalchemy import text

    from app.db.session import get_engine
    from app.memory.vectorstore import get_chroma_client

    components: dict[str, str] = {}
    try:
        with get_engine().connect() as conn:
            conn.execute(text("select 1"))
        components["postgres"] = "ok"
    except Exception as exc:
        components["postgres"] = f"error: {exc.__class__.__name__}"
    try:
        get_chroma_client().heartbeat()
        components["chroma"] = "ok"
    except Exception as exc:
        components["chroma"] = f"error: {exc.__class__.__name__}"
    status = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    components["langsmith_tracing"] = "enabled" if os.environ.get("LANGSMITH_TRACING") == "true" else "disabled"
    return HealthOut(status=status, components=components)
