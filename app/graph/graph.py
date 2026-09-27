"""Orchestrator: the LangGraph StateGraph that runs one conversation turn.

    START -> load_context -> query_understanding
        chitchat / out_of_scope ------------------------------> context_synthesis
        needs_history ---------> query_rewriting --+
        multi_part ------------------------------------> query_decomposition
        otherwise -------------------------------------------------+
                                                                   v
        retrieval_router -> retrieval -> context_synthesis
    context_synthesis -> [summarization if the thread is long] -> memory_management -> END

State is checkpointed per thread_id (session) in PostgreSQL after every node,
and long-term memories live in the PostgreSQL store, namespaced per user.
"""

import uuid
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from app.graph.nodes.context_synthesis import synthesize_answer
from app.graph.nodes.load_context import load_context
from app.graph.nodes.memory_management import manage_memory
from app.graph.nodes.query_decomposition import decompose_query
from app.graph.nodes.query_rewriting import rewrite_query
from app.graph.nodes.query_understanding import understand_query
from app.graph.nodes.retrieval import retrieval_node
from app.graph.nodes.retrieval_router import route_retrieval
from app.graph.nodes.summarization import needs_summarization, summarize_conversation
from app.graph.state import ConversationState

LOAD_CONTEXT = "load_context"
UNDERSTANDING = "query_understanding"
REWRITING = "query_rewriting"
DECOMPOSITION = "query_decomposition"
ROUTER = "retrieval_router"
RETRIEVAL = "retrieval"
SYNTHESIS = "context_synthesis"
SUMMARIZATION = "summarization"
MEMORY = "memory_management"


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #

def route_after_understanding(state: ConversationState) -> str:
    if state.get("intent") in ("chitchat", "out_of_scope"):
        return SYNTHESIS
    if state.get("needs_history"):
        return REWRITING
    if state.get("intent") == "multi_part":
        return DECOMPOSITION
    return ROUTER


def route_after_rewriting(state: ConversationState) -> str:
    return DECOMPOSITION if state.get("intent") == "multi_part" else ROUTER


def route_after_answer(state: ConversationState) -> str:
    return SUMMARIZATION if needs_summarization(state) else MEMORY


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def build_graph(checkpointer: BaseCheckpointSaver | None = None, store: BaseStore | None = None) -> CompiledStateGraph:
    builder = StateGraph(ConversationState)
    builder.add_node(LOAD_CONTEXT, load_context)
    builder.add_node(UNDERSTANDING, understand_query)
    builder.add_node(REWRITING, rewrite_query)
    builder.add_node(DECOMPOSITION, decompose_query)
    builder.add_node(ROUTER, route_retrieval)
    builder.add_node(RETRIEVAL, retrieval_node)
    builder.add_node(SYNTHESIS, synthesize_answer)
    builder.add_node(SUMMARIZATION, summarize_conversation)
    builder.add_node(MEMORY, manage_memory)

    builder.add_edge(START, LOAD_CONTEXT)
    builder.add_edge(LOAD_CONTEXT, UNDERSTANDING)
    builder.add_conditional_edges(UNDERSTANDING, route_after_understanding, [SYNTHESIS, REWRITING, DECOMPOSITION, ROUTER])
    builder.add_conditional_edges(REWRITING, route_after_rewriting, [DECOMPOSITION, ROUTER])
    builder.add_edge(DECOMPOSITION, ROUTER)
    builder.add_edge(ROUTER, RETRIEVAL)
    builder.add_edge(RETRIEVAL, SYNTHESIS)
    builder.add_conditional_edges(SYNTHESIS, route_after_answer, [SUMMARIZATION, MEMORY])
    builder.add_edge(SUMMARIZATION, MEMORY)
    builder.add_edge(MEMORY, END)
    return builder.compile(checkpointer=checkpointer, store=store)


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """The application graph, persisted in PostgreSQL (checkpoints + long-term store)."""
    from app.memory.checkpointer import get_checkpointer
    from app.memory.long_term import get_store
    from app.observability.tracing import configure_langsmith

    configure_langsmith()
    return build_graph(checkpointer=get_checkpointer(), store=get_store())


# --------------------------------------------------------------------------- #
# Running a turn
# --------------------------------------------------------------------------- #

def turn_input(message: str, user_id: str, thread_id: str) -> tuple[dict, dict]:
    """Graph input and config for one user message in a session.

    `run_id` becomes the LangSmith trace id of the turn; `session_id` metadata
    groups a session's turns into one thread in LangSmith.
    """
    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "run_id": uuid.uuid4(),
        "run_name": "conversation_turn",
        "tags": ["conversational-rag"],
        "metadata": {"session_id": thread_id, "user_id": user_id},
    }
    return {"messages": [HumanMessage(message)], "user_id": user_id, "thread_id": thread_id}, config


def run_turn(graph: CompiledStateGraph, message: str, user_id: str, thread_id: str) -> dict[str, Any]:
    """Run one turn to completion and return the final state."""
    inputs, config = turn_input(message, user_id, thread_id)
    return graph.invoke(inputs, config)


def mermaid_diagram() -> str:
    """Mermaid source for the workflow (used in the README / docs)."""
    return build_graph().get_graph().draw_mermaid()
