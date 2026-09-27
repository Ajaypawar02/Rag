"""Test doubles shared by graph and API tests: a scripted LLM and an in-memory retriever."""

import json

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import app.llm as llm_module
from app.graph.nodes import retrieval

CATALOG = (
    {"source": "fastapi_dependencies.md", "product": "fastapi", "version": None, "title": "Dependencies"},
    {"source": "pydantic_v2_migration_guide.md", "product": "pydantic", "version": "2", "title": "Migration"},
    {"source": "docker_compose_getting_started.md", "product": "docker", "version": None, "title": "Compose"},
)

NO_MEMORY = {"memories": [], "reasoning": "nothing durable"}


def understanding(intent: str, needs_history: bool = False) -> dict:
    return {"intent": intent, "needs_history": needs_history, "reasoning": "test"}


def plan(strategy: str = "hybrid", **filters) -> dict:
    return {"strategy": strategy, "filters": filters, "reasoning": "test"}


class ScriptedLLM:
    """Replaces app.llm.get_llm; each model call consumes the next scripted reply.

    The fake model has no native structured output, so structured calls go
    through the JSON fallback in `invoke_structured` (one reply each).
    """

    def __init__(self, monkeypatch):
        self._model = None
        monkeypatch.setattr(llm_module, "get_llm", lambda fast=False: self._model)

    def script(self, *replies) -> None:
        texts = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self._model = GenericFakeChatModel(messages=iter(AIMessage(t) for t in texts))

    def assert_consumed(self) -> None:
        with pytest.raises(StopIteration):
            next(self._model.messages)


def make_fake_retrieve(calls: list):
    """A `retrieve()` stand-in returning one FastAPI chunk and recording its arguments."""

    def fake_retrieve(query, strategy="hybrid", filters=None, sub_queries=None, **_):
        calls.append({"query": query, "strategy": strategy, "filters": filters, "sub_queries": sub_queries})
        doc = Document(page_content=f"Docs about {query}", metadata={
            "chunk_id": "fastapi_dependencies.md::0", "source": "fastapi_dependencies.md",
            "title": "Dependencies", "section": "First Steps"})
        result = retrieval.RetrievalResult(query=query, strategy=strategy, filters=filters or {},
                                           sub_queries=sub_queries or [])
        result.chunks = [retrieval.RetrievedChunk(document=doc, score=0.9)]
        result.timings_ms = {"total": 1.0}
        return result

    return fake_retrieve
