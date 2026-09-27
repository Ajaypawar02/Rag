"""Shared fixtures: scripted LLM and in-memory retrieval (no provider, Chroma or Postgres needed)."""

import pytest

from app.graph.nodes import context_synthesis, query_understanding, retrieval, retrieval_router
from tests.helpers import CATALOG, ScriptedLLM, make_fake_retrieve


@pytest.fixture
def fake_llm(monkeypatch) -> ScriptedLLM:
    return ScriptedLLM(monkeypatch)


@pytest.fixture
def fake_retrieval(monkeypatch) -> list:
    """Patch the document catalog and retrieve(); returns the list of recorded retrieve() calls."""
    for module in (query_understanding, retrieval_router, context_synthesis):
        monkeypatch.setattr(module, "get_catalog", lambda: CATALOG)
    calls: list = []
    monkeypatch.setattr(retrieval, "retrieve", make_fake_retrieve(calls))
    return calls
