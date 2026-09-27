"""Tests for graph state, routing, agent nodes and full multi-turn runs.

A scripted fake LLM stands in for the provider: it answers each model call in
order. It does not support native structured output, so these runs also
exercise the JSON fallback path in `invoke_structured`.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from app.config import get_settings
from app.graph import graph as graph_module
from app.graph.graph import build_graph, route_after_understanding, run_turn
from app.graph.nodes.context_synthesis import cited_sources
from app.graph.nodes.retrieval_router import RouterFilters, validate_filters
from app.graph.state import ConversationState, latest_user_message, reduce_decisions, turn_timings
from app.llm import StructuredOutputError, invoke_structured
from app.memory.long_term import user_namespace
from tests.helpers import CATALOG, NO_MEMORY, plan, understanding


@pytest.fixture
def app_graph():
    store = InMemoryStore()
    return build_graph(checkpointer=InMemorySaver(), store=store), store


# --------------------------------------------------------------------------- #
# State and helpers
# --------------------------------------------------------------------------- #

def _decision(agent: str, turn: int) -> dict:
    return {"agent": agent, "turn": turn, "decision": "ok", "reasoning": "", "details": {}, "latency_ms": 2.0}


def test_decisions_reset_when_a_new_turn_starts():
    log = reduce_decisions(None, [_decision("a", 1)])
    log = reduce_decisions(log, [_decision("b", 1)])
    assert [d["agent"] for d in log] == ["a", "b"]
    log = reduce_decisions(log, [_decision("c", 2)])
    assert [d["agent"] for d in log] == ["c"]
    assert reduce_decisions(log, None) == log
    assert turn_timings({"decisions": log + [_decision("c", 2)]}) == {"c": 4.0}


def test_latest_user_message():
    state = {"messages": [HumanMessage("first"), AIMessage("reply"), HumanMessage("second")]}
    assert latest_user_message(state) == "second"
    assert latest_user_message({"messages": []}) == ""


def test_route_after_understanding():
    assert route_after_understanding({"intent": "chitchat"}) == "context_synthesis"
    assert route_after_understanding({"intent": "follow_up", "needs_history": True}) == "query_rewriting"
    assert route_after_understanding({"intent": "multi_part"}) == "query_decomposition"
    assert route_after_understanding({"intent": "new_question"}) == "retrieval_router"


def test_validate_filters_against_catalog():
    assert validate_filters(RouterFilters(product="Pydantic", version="2", has_code=True), CATALOG)[0] == {
        "product": "pydantic", "version": "2", "has_code": True}
    filters, notes = validate_filters(RouterFilters(product="kafka", version="3"), CATALOG)
    assert filters == {} and len(notes) == 2
    filters, notes = validate_filters(RouterFilters(product="fastapi", version="9"), CATALOG)
    assert filters == {"product": "fastapi"} and "dropped version" in notes[0]
    # A specific document wins over product/version filters.
    assert validate_filters(RouterFilters(source="fastapi_dependencies.md", product="fastapi"), CATALOG)[0] == {
        "source": "fastapi_dependencies.md"}
    filters, notes = validate_filters(RouterFilters(source="missing.pdf"), CATALOG)
    assert filters == {} and "unknown source" in notes[0]


def test_uploaded_documents_are_visible_to_agents():
    from app.graph import prompts

    catalog = CATALOG + ({"source": "dt_report.html", "title": "Proactive Digital Twin", "product": None,
                          "origin": "upload", "ingested_at": "2026-09-27T15:17:26+00:00"},)
    assert prompts.catalog_topics(catalog)[-1] == '"Proactive Digital Twin"'
    table = prompts.render_catalog(catalog).splitlines()
    assert table[1].startswith("dt_report.html | Proactive Digital Twin")  # newest upload listed first


def test_cited_sources_maps_markers_in_order():
    retrieved = [{"source": "a.md", "section": "S1"}, {"source": "b.md", "page": 3}]
    assert cited_sources("See [2] and [1], again [2]; ignore [7].", retrieved) == [
        {"index": 2, "source": "b.md", "page": 3},
        {"index": 1, "source": "a.md", "section": "S1"},
    ]


def test_normalize_citations():
    from app.graph.nodes.context_synthesis import normalize_citations

    assert normalize_citations("See [5]【5】 and 【2†source】, also [3] [3].") == "See [5] and [2], also [3]."


class _Answer(BaseModel):
    value: int


def test_invoke_structured_json_fallback_and_error(fake_llm):
    fake_llm.script('Sure! ```json\n{"value": 3}\n```')
    assert invoke_structured(_Answer, [HumanMessage("x")]).value == 3
    fake_llm.script("I cannot do that")
    with pytest.raises(StructuredOutputError):
        invoke_structured(_Answer, [HumanMessage("x")])


# --------------------------------------------------------------------------- #
# Full conversations
# --------------------------------------------------------------------------- #

def test_multi_turn_conversation(fake_llm, fake_retrieval, app_graph):
    graph, store = app_graph

    # Turn 1: standalone question, routed with a product filter.
    fake_llm.script(understanding("new_question"), plan(product="fastapi"),
                    "Use `Depends` in the parameter [1].", NO_MEMORY)
    out = run_turn(graph, "How do I declare a dependency in FastAPI?", "alice", "s1")
    assert out["turn"] == 1 and out["intent"] == "new_question"
    assert fake_retrieval[-1] == {"query": "How do I declare a dependency in FastAPI?", "strategy": "hybrid",
                                  "filters": {"product": "fastapi"}, "sub_queries": None}
    assert out["answer"].startswith("Use `Depends`")
    assert out["sources"] == [{"index": 1, "source": "fastapi_dependencies.md", "title": "Dependencies",
                               "section": "First Steps"}]
    assert [d["agent"] for d in out["decisions"]] == [
        "load_context", "query_understanding", "retrieval_router", "retrieval", "context_synthesis",
        "memory_management"]
    fake_llm.assert_consumed()

    # Turn 2: follow-up is rewritten using history; a durable fact is remembered.
    fake_llm.script(understanding("follow_up", True),
                    {"standalone_query": "How to test FastAPI dependencies with pytest", "reasoning": "resolved 'it'"},
                    plan(), "Override it with `app.dependency_overrides` [1].",
                    {"memories": [{"content": "Uses pytest for testing", "category": "tech_stack"}],
                     "reasoning": "testing tool"})
    out = run_turn(graph, "How do I test it? I use pytest.", "alice", "s1")
    assert out["turn"] == 2 and out["needs_history"] is True
    assert fake_retrieval[-1]["query"] == "How to test FastAPI dependencies with pytest"
    assert out["memory_updates"] == ["Uses pytest for testing"]
    assert [i.value["text"] for i in store.search(user_namespace("alice"))] == ["Uses pytest for testing"]
    assert {d["turn"] for d in out["decisions"]} == {2}  # log holds only the current turn
    assert len(out["messages"]) == 4

    # Turn 3: chitchat skips retrieval; short message skips the memory model call.
    fake_llm.script(understanding("chitchat"), "You're welcome!")
    out = run_turn(graph, "thanks!", "alice", "s1")
    assert out["answer"] == "You're welcome!" and out["retrieved"] == [] and len(fake_retrieval) == 2
    fake_llm.assert_consumed()

    # Another session of the same user starts clean but sees the long-term memory.
    fake_llm.script(understanding("new_question"), plan(), "Answer [1].", NO_MEMORY)
    out = run_turn(graph, "What is dependency injection in FastAPI?", "alice", "s2")
    assert out["turn"] == 1 and len(out["messages"]) == 2
    assert out["user_memories"] == ["Uses pytest for testing"]

    # A different user sees neither the session nor the memory.
    fake_llm.script(understanding("new_question"), plan(), "Answer [1].", NO_MEMORY)
    out = run_turn(graph, "What is dependency injection in FastAPI?", "bob", "s3")
    assert out["user_memories"] == []


def test_multi_part_question_is_decomposed(fake_llm, fake_retrieval, app_graph):
    graph, _ = app_graph
    fake_llm.script(
        understanding("multi_part"),
        {"sub_queries": ["How to declare FastAPI dependencies", "How to start services with Docker Compose"],
         "reasoning": "two topics"},
        plan(product="fastapi"),
        "First [1]. Then [1].",
        NO_MEMORY,
    )
    out = run_turn(graph, "How do I add a FastAPI dependency and run it with Docker Compose?", "u", "t")
    assert out["retrieval_strategy"] == "multi_query"
    assert fake_retrieval[-1]["sub_queries"] == [
        "How to declare FastAPI dependencies", "How to start services with Docker Compose"]


def test_model_failures_fall_back_without_breaking_the_turn(fake_llm, fake_retrieval, app_graph):
    graph, _ = app_graph
    # understanding, router, (answer), memory: each structured call gets unparseable text.
    fake_llm.script("garbage", "garbage", "Still answered [1].", "garbage")
    out = run_turn(graph, "How do I declare a dependency in FastAPI?", "u", "t")
    assert out["answer"] == "Still answered [1]."
    fallbacks = {d["agent"] for d in out["decisions"] if d["details"].get("fallback")}
    assert fallbacks == {"query_understanding", "retrieval_router"}
    assert fake_retrieval[-1]["strategy"] == "hybrid"


def test_long_conversation_is_summarized_and_pruned(fake_llm, fake_retrieval, app_graph, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "summary_threshold", 3)
    monkeypatch.setattr(s, "window_size", 2)
    monkeypatch.setattr(s, "memory_strategy", "summary_buffer")
    graph, _ = app_graph

    fake_llm.script(understanding("new_question"), plan(), "A1 [1].", NO_MEMORY)
    run_turn(graph, "First question about FastAPI dependencies?", "u", "t")
    fake_llm.script(understanding("new_question"), plan(), "A2 [1].",
                    "User asked about FastAPI dependencies; learned to use Depends.", NO_MEMORY)
    out = run_turn(graph, "Second question about Docker Compose?", "u", "t")

    assert out["summary"].startswith("User asked about FastAPI dependencies")
    assert [m.content for m in out["messages"]] == ["Second question about Docker Compose?", "A2 [1]."]
    summary_decision = next(d for d in out["decisions"] if d["agent"] == "summarization")
    assert summary_decision["details"]["pruned_messages"] == 2


def test_sliding_window_drops_without_summary(fake_llm, fake_retrieval, app_graph, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "summary_threshold", 3)
    monkeypatch.setattr(s, "window_size", 2)
    monkeypatch.setattr(s, "memory_strategy", "sliding_window")
    graph, _ = app_graph

    fake_llm.script(understanding("new_question"), plan(), "A1 [1].", NO_MEMORY)
    run_turn(graph, "First question about FastAPI dependencies?", "u", "t")
    fake_llm.script(understanding("new_question"), plan(), "A2 [1].", NO_MEMORY)
    out = run_turn(graph, "Second question about Docker Compose?", "u", "t")
    assert out.get("summary", "") == "" and len(out["messages"]) == 2


def test_turn_config_carries_tracing_metadata():
    inputs, config = graph_module.turn_input("hi", "alice", "s1")
    assert config["configurable"] == {"thread_id": "s1", "user_id": "alice"}
    assert config["metadata"] == {"session_id": "s1", "user_id": "alice"}
    assert config["run_name"] == "conversation_turn" and config["run_id"] is not None
    assert graph_module.turn_input("hi", "alice", "s1")[1]["run_id"] != config["run_id"]


def test_configure_langsmith_exports_settings(monkeypatch):
    from app.observability import tracing

    s = get_settings()
    for var in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(s, "langsmith_tracing", True)
    monkeypatch.setattr(s, "langsmith_api_key", None)
    tracing.configure_langsmith.cache_clear()
    assert tracing.configure_langsmith() is False  # enabled but no key

    from pydantic import SecretStr

    monkeypatch.setattr(s, "langsmith_api_key", SecretStr("lsv2_test"))
    monkeypatch.setattr(s, "langsmith_project", "rag-tests")
    tracing.configure_langsmith.cache_clear()
    assert tracing.configure_langsmith() is True
    import os

    assert os.environ["LANGSMITH_TRACING"] == "true" and os.environ["LANGSMITH_PROJECT"] == "rag-tests"
    tracing.configure_langsmith.cache_clear()
    monkeypatch.setenv("LANGSMITH_TRACING", "false")  # don't leak tracing into other tests


def test_mermaid_diagram_lists_all_agents():
    diagram = graph_module.mermaid_diagram()
    for node in ("query_understanding", "query_rewriting", "query_decomposition", "retrieval_router",
                 "retrieval", "context_synthesis", "summarization", "memory_management"):
        assert node in diagram


def test_state_isolated_per_thread_with_plain_graph():
    """Sanity check of LangGraph checkpointing semantics the app relies on."""

    def echo(state: ConversationState) -> dict:
        return {"messages": [AIMessage("ok")]}

    builder = StateGraph(ConversationState)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    g = builder.compile(checkpointer=InMemorySaver())
    g.invoke({"messages": [HumanMessage("a")]}, {"configurable": {"thread_id": "1"}})
    out = g.invoke({"messages": [HumanMessage("b")]}, {"configurable": {"thread_id": "1"}})
    other = g.invoke({"messages": [HumanMessage("c")]}, {"configurable": {"thread_id": "2"}})
    assert len(out["messages"]) == 4 and len(other["messages"]) == 2
