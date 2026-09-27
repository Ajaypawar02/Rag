"""Memory Management Agent: decides what to store in long-term, cross-session memory.

Looks at the user's latest message for durable facts (preferences, stack,
expertise, goals), skips anything already known, and saves the rest to the
user's namespace in the store. Very short messages ("thanks", "ok") are
skipped without a model call.
"""

from typing import Literal

from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from app import llm
from app.graph import prompts
from app.graph.state import ConversationState
from app.memory.long_term import save_memory, search_memories
from app.observability.decision_log import decision, traced

MIN_WORDS_TO_ANALYZE = 4
DUPLICATE_SIMILARITY = 0.9  # a new fact this similar to an existing one is not stored again


class MemoryItem(BaseModel):
    content: str = Field(description="A durable fact about the user, in the third person.")
    category: Literal["preference", "tech_stack", "expertise", "goal"]


class MemoryDecision(BaseModel):
    memories: list[MemoryItem] = Field(default_factory=list, description="Facts to store; usually empty.")
    reasoning: str = Field(description="One sentence on why these facts (or none) were chosen.")


def _is_duplicate(store: BaseStore, user_id: str, text: str, known: list[str]) -> bool:
    if text.strip().lower() in {k.strip().lower() for k in known}:
        return True
    nearest = search_memories(store, user_id, text, limit=1)
    return bool(nearest) and (nearest[0].get("score") or 0) >= DUPLICATE_SIMILARITY


@traced("memory_management")
def manage_memory(state: ConversationState, *, store: BaseStore | None = None) -> dict:
    query = state.get("query", "")
    if store is None:
        return {"decisions": [decision("skipped", "No long-term store configured.")]}
    if len(query.split()) < MIN_WORDS_TO_ANALYZE:
        return {"decisions": [decision("skipped", "Message too short to contain durable facts.")]}

    user_id = state.get("user_id", "anonymous")
    known = state.get("user_memories", [])
    try:
        result = llm.invoke_structured(MemoryDecision, prompts.memory_messages(query, known), fast=True)
    except Exception as exc:
        return {"decisions": [decision("skipped", f"Model error: {exc}")]}

    stored, skipped = [], []
    for item in result.memories:
        if _is_duplicate(store, user_id, item.content, known):
            skipped.append(item.content)
            continue
        save_memory(store, user_id, item.content, item.category)
        stored.append(item.content)

    summary = f"stored {len(stored)} memories" + (f", skipped {len(skipped)} duplicates" if skipped else "")
    return {
        "memory_updates": stored,
        "decisions": [decision(summary, result.reasoning, stored=stored, duplicates=skipped)],
    }
