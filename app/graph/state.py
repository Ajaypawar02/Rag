"""Shared state for the conversational RAG graph.

Every node receives the full `ConversationState` and returns a partial dict
of the keys it changes; LangGraph merges that update into the state and the
Postgres checkpointer persists it per `thread_id` (one conversation session).

Keys fall into three groups:

- Conversation (persist across turns): messages, summary, identity.
- Per-turn working data (reset by `turn_reset()` at the start of each turn):
  intent, rewritten query, retrieval plan and results, answer.
- Per-turn observability: the agent decision log (what each agent decided,
  why, and how long it took) behind the explainability dashboard and latency
  metrics. Entries carry the turn number and older turns are dropped
  automatically, so checkpoints do not grow with conversation length.

All values are JSON-friendly (plain dicts/lists, no Document objects) so
checkpoints stay small and serializable.
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.graph.nodes.retrieval import Strategy

Intent = Literal[
    "new_question",  # standalone technical question
    "follow_up",  # depends on earlier turns ("how do I test it?")
    "multi_part",  # several questions at once -> query decomposition
    "chitchat",  # greetings, thanks; no retrieval needed
    "out_of_scope",  # unrelated to the documentation
]


class AgentDecision(TypedDict):
    """One entry in the per-turn decision log (explainability)."""

    turn: int
    agent: str  # node name, e.g. "query_understanding"
    decision: str  # short outcome, e.g. "follow_up, needs_history=True"
    reasoning: str  # why, usually from the LLM's structured output
    details: dict[str, Any]
    latency_ms: float


class RetrievedContext(TypedDict, total=False):
    """A retrieved chunk as passed to the answer agent and shown as a source."""

    chunk_id: str
    content: str
    source: str
    title: str
    section: str
    page: int
    score: float


class Source(TypedDict, total=False):
    """A citation attached to the final answer."""

    index: int  # the [n] marker used in the answer text
    source: str
    title: str
    section: str
    page: int


def reduce_decisions(current: list[AgentDecision] | None, update: list[AgentDecision] | None) -> list[AgentDecision]:
    """Append decisions within a turn; the first decision of a new turn starts a fresh log."""
    if not update:
        return current or []
    turn = update[0].get("turn")
    return [d for d in (current or []) if d.get("turn") == turn] + update


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class ConversationState(TypedDict, total=False):
    # --- Conversation (persisted across turns) -------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    thread_id: str
    turn: int  # number of user turns so far in this thread
    summary: str  # rolling summary of messages pruned from `messages`
    user_memories: list[str]  # long-term facts about the user, loaded each turn

    # --- Query understanding / rewriting -------------------------------------
    query: str  # the latest user message, verbatim
    intent: Intent
    needs_history: bool
    rewritten_query: str  # standalone query used for retrieval
    sub_queries: list[str]  # from query decomposition (multi_part)

    # --- Retrieval ------------------------------------------------------------
    retrieval_strategy: Strategy
    filters: dict[str, Any]  # metadata filters chosen by the router
    retrieved: list[RetrievedContext]
    retrieval_info: dict[str, Any]  # RetrievalResult.to_dict() diagnostics

    # --- Answer ---------------------------------------------------------------
    answer: str
    sources: list[Source]

    # --- Memory management ----------------------------------------------------
    memory_updates: list[str]  # facts stored to long-term memory this turn

    # --- Observability (current turn only) -----------------------------------
    decisions: Annotated[list[AgentDecision], reduce_decisions]


def turn_reset() -> dict[str, Any]:
    """State update that clears all per-turn keys; applied at the start of every turn."""
    return {
        "intent": "new_question",
        "needs_history": False,
        "rewritten_query": "",
        "sub_queries": [],
        "retrieval_strategy": "hybrid",
        "filters": {},
        "retrieved": [],
        "retrieval_info": {},
        "answer": "",
        "sources": [],
        "memory_updates": [],
    }


def turn_timings(state: ConversationState) -> dict[str, float]:
    """Per-agent latency (ms) for the current turn, derived from the decision log."""
    timings: dict[str, float] = {}
    for d in state.get("decisions", []):
        timings[d["agent"]] = timings.get(d["agent"], 0.0) + d["latency_ms"]
    return timings


def latest_user_message(state: ConversationState) -> str:
    """Text of the most recent human message ('' if there is none)."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.text)  # joins text blocks when content is a list
    return ""
