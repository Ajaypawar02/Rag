"""API tests: sessions, isolation, chat (blocking + SSE), history, decisions, memories, documents, metrics.

Runs against in-memory SQLite, an in-memory LangGraph checkpointer/store and the
scripted LLM, so no Postgres, Chroma or provider is needed.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import deps
from app.api.routes import documents as documents_route
from app.api.routes import metrics as metrics_route
from app.config import get_settings
from app.db.session import init_db
from app.graph.graph import build_graph
from app.ingestion.pipeline import IngestResult
from app.main import app
from tests.helpers import NO_MEMORY, plan, understanding

ALICE = {"X-User-Id": "alice"}
BOB = {"X-User-Id": "bob"}


@pytest.fixture
def client(fake_llm, fake_retrieval, monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    init_db(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    graph = build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())

    app.dependency_overrides[deps.get_sessionmaker] = lambda: factory
    app.dependency_overrides[deps.get_graph] = lambda: graph
    monkeypatch.setattr(metrics_route, "memory_overhead", lambda: {"state_bytes_total": 0})
    yield TestClient(app)  # no `with`: skips the startup warm-up that connects to real services
    app.dependency_overrides.clear()


def new_session(client: TestClient, headers=ALICE) -> str:
    response = client.post("/sessions", headers=headers, json={})
    assert response.status_code == 201
    return response.json()["session_id"]


def answer_turn(fake_llm, answer="Use `Depends` [1].", memory=NO_MEMORY, intent="new_question", **plan_filters):
    fake_llm.script(understanding(intent), plan(**plan_filters), answer, memory)


# --- Sessions and isolation --------------------------------------------------

def test_user_header_is_required_and_validated(client):
    assert client.post("/sessions", json={}).status_code == 422
    assert client.post("/sessions", headers={"X-User-Id": "bad id!"}, json={}).status_code == 400


def test_sessions_are_isolated_per_user(client):
    sid = new_session(client)
    assert [s["session_id"] for s in client.get("/sessions", headers=ALICE).json()] == [sid]
    assert client.get("/sessions", headers=BOB).json() == []
    assert client.get(f"/sessions/{sid}", headers=BOB).status_code == 404
    assert client.get(f"/sessions/{sid}/history", headers=BOB).status_code == 404
    assert client.post(f"/sessions/{sid}/chat", headers=BOB, json={"message": "hi"}).status_code == 404


def test_create_session_with_title(client):
    body = client.post("/sessions", headers=ALICE, json={"title": "K8s questions"}).json()
    assert body["title"] == "K8s questions" and body["turns"] == 0


# --- Chat ---------------------------------------------------------------------

def test_chat_history_decisions_and_metrics(client, fake_llm, fake_retrieval):
    sid = new_session(client)
    answer_turn(fake_llm, product="fastapi")
    body = client.post(f"/sessions/{sid}/chat", headers=ALICE,
                       json={"message": "How do I declare a dependency in FastAPI?"}).json()
    assert body["answer"] == "Use `Depends` [1]." and body["turn"] == 1
    assert body["sources"][0]["source"] == "fastapi_dependencies.md"
    assert body["filters"] == {"product": "fastapi"} and body["retrieval_strategy"] == "hybrid"
    assert set(body["agent_latency_ms"]) >= {"query_understanding", "retrieval_router", "context_synthesis"}
    assert len(body["trace_id"]) == 36  # LangSmith root run id (uuid)

    # Follow-up uses the session history via the rewriting agent.
    fake_llm.script(understanding("follow_up", True),
                    {"standalone_query": "How to test FastAPI dependencies", "reasoning": "resolved it"},
                    plan(), "Override it [1].", NO_MEMORY)
    body = client.post(f"/sessions/{sid}/chat", headers=ALICE, json={"message": "How do I test it?"}).json()
    assert body["turn"] == 2 and body["rewritten_query"] == "How to test FastAPI dependencies"

    history = client.get(f"/sessions/{sid}/history", headers=ALICE).json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant", "user", "assistant"]

    transcript = client.get(f"/sessions/{sid}/transcript", headers=ALICE).json()
    assert [(t["turn"], t["answer"]) for t in transcript] == [(1, "Use `Depends` [1]."), (2, "Override it [1].")]
    assert transcript[0]["sources"][0]["source"] == "fastapi_dependencies.md"

    session = client.get(f"/sessions/{sid}", headers=ALICE).json()
    assert session["turns"] == 2 and session["title"] == "How do I declare a dependency in FastAPI?"

    traces = client.get(f"/sessions/{sid}/decisions", headers=ALICE).json()
    assert [t["turn"] for t in traces] == [2, 1]
    assert traces[0]["trace_id"] == body["trace_id"]
    assert "query_rewriting" in [d["agent"] for d in traces[0]["decisions"]]
    assert len(client.get(f"/sessions/{sid}/decisions?turn=1", headers=ALICE).json()) == 1

    metrics = client.get("/metrics").json()
    assert metrics["turns"] == 2 and metrics["intents"] == {"new_question": 1, "follow_up": 1}
    assert metrics["citation_rate"] == 1.0 and metrics["latency_ms"]["p50"] is not None


def test_chat_stream_emits_agents_tokens_and_done(client, fake_llm):
    sid = new_session(client)
    answer_turn(fake_llm)
    with client.stream("POST", f"/sessions/{sid}/chat/stream", headers=ALICE,
                       json={"message": "How do I declare a dependency in FastAPI?"}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

    events = [(block.split("\n")[0].removeprefix("event: "), json.loads(block.split("\n")[1].removeprefix("data: ")))
              for block in raw.strip().split("\n\n")]
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "done" and "token" in kinds
    assert "".join(data["text"] for kind, data in events if kind == "token") == "Use `Depends` [1]."
    assert {data["agent"] for kind, data in events if kind == "agent"} >= {"query_understanding", "context_synthesis"}
    assert events[-1][1]["answer"] == "Use `Depends` [1]."

    trace = client.get(f"/sessions/{sid}/decisions", headers=ALICE).json()[0]
    assert trace["first_token_ms"] is not None


def test_concurrent_turn_on_same_session_is_rejected(client, fake_llm):
    sid = new_session(client)
    release = deps.acquire_session_turn(sid)
    try:
        response = client.post(f"/sessions/{sid}/chat", headers=ALICE, json={"message": "hello there"})
        assert response.status_code == 409
    finally:
        release()


def test_chat_failure_returns_503(client):
    class BrokenGraph:
        checkpointer = store = None

        def invoke(self, *_):
            raise RuntimeError("vector store down")

    app.dependency_overrides[deps.get_graph] = lambda: BrokenGraph()
    sid = new_session(client)
    response = client.post(f"/sessions/{sid}/chat", headers=ALICE, json={"message": "hello there"})
    assert response.status_code == 503 and "vector store down" in response.json()["detail"]


def test_delete_session_removes_history(client, fake_llm):
    sid = new_session(client)
    answer_turn(fake_llm)
    client.post(f"/sessions/{sid}/chat", headers=ALICE, json={"message": "How do I declare a dependency?"})
    assert client.delete(f"/sessions/{sid}", headers=BOB).status_code == 404
    assert client.delete(f"/sessions/{sid}", headers=ALICE).status_code == 204
    assert client.get(f"/sessions/{sid}", headers=ALICE).status_code == 404


# --- Long-term memory ----------------------------------------------------------

def test_memories_are_listed_and_forgotten_per_user(client, fake_llm):
    sid = new_session(client)
    answer_turn(fake_llm, memory={"memories": [{"content": "Uses Pydantic v2", "category": "tech_stack"}],
                                  "reasoning": "stated stack"})
    body = client.post(f"/sessions/{sid}/chat", headers=ALICE,
                       json={"message": "I use Pydantic v2 - how do validators work?"}).json()
    assert body["memory_updates"] == ["Uses Pydantic v2"]

    memories = client.get("/memories", headers=ALICE).json()
    assert [m["text"] for m in memories] == ["Uses Pydantic v2"]
    assert client.get("/memories", headers=BOB).json() == []
    key = memories[0]["key"]
    assert client.delete(f"/memories/{key}", headers=BOB).status_code == 404
    assert client.delete(f"/memories/{key}", headers=ALICE).status_code == 204
    assert client.get("/memories", headers=ALICE).json() == []


# --- Documents ---------------------------------------------------------------

@pytest.fixture
def upload_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path)
    return tmp_path


def test_upload_document(client, upload_dir, monkeypatch):
    seen = {}

    def fake_ingest(path, origin):
        seen.update(path=path, origin=origin)
        return IngestResult(source=path.name, title="Guide", doc_type="markdown", doc_category="guide",
                            product=None, version=None, chunks=3, chunks_with_code=1, replaced_chunks=0, seconds=0.1)

    monkeypatch.setattr(documents_route, "ingest_file", fake_ingest)
    response = client.post("/documents/upload", files={"file": ("../My Guide.md", b"# Guide\ntext", "text/markdown")})
    assert response.status_code == 201 and response.json()["chunks"] == 3
    assert seen["origin"] == "upload" and seen["path"] == upload_dir / "My_Guide.md"
    assert (upload_dir / "My_Guide.md").read_bytes() == b"# Guide\ntext"


def test_upload_rejects_bad_files(client, upload_dir):
    assert client.post("/documents/upload", files={"file": ("notes.docx", b"x")}).status_code == 415
    assert client.post("/documents/upload", files={"file": ("empty.md", b"")}).status_code == 422


def test_list_and_delete_documents(client, monkeypatch, upload_dir):
    monkeypatch.setattr(documents_route, "list_documents", lambda: [{"source": "a.md", "chunks": 2}])
    assert client.get("/documents").json()[0]["source"] == "a.md"
    monkeypatch.setattr(documents_route, "remove_document", lambda source: 2 if source == "a.md" else 0)
    assert client.delete("/documents/a.md").json() == {"deleted": "a.md", "chunks_removed": 2}
    assert client.delete("/documents/missing.md").status_code == 404


def test_openapi_documents_all_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/documents/upload", "/documents", "/sessions", "/sessions/{session_id}/chat",
                 "/sessions/{session_id}/chat/stream", "/sessions/{session_id}/history",
                 "/sessions/{session_id}/decisions", "/memories", "/metrics", "/health"):
        assert path in paths
