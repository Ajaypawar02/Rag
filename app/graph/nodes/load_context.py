"""Load Context node: starts every turn.

Resets per-turn state, increments the turn counter, captures the user's
message, and loads the long-term memories relevant to it from the store.
"""

from langgraph.store.base import BaseStore

from app.graph.state import ConversationState, latest_user_message, turn_reset
from app.memory.long_term import search_memories
from app.observability.decision_log import decision, traced

MEMORIES_PER_TURN = 5


@traced("load_context")
def load_context(state: ConversationState, *, store: BaseStore | None = None) -> dict:
    query = latest_user_message(state)
    turn = state.get("turn", 0) + 1
    memories = search_memories(store, state.get("user_id", "anonymous"), query, MEMORIES_PER_TURN) if store else []
    texts = [m["text"] for m in memories]
    return {
        **turn_reset(),
        "turn": turn,
        "query": query,
        "user_memories": texts,
        "decisions": [
            decision(
                f"turn {turn}: loaded {len(texts)} long-term memories",
                "Memories ranked by relevance to the current message.",
                memories=texts,
                history_messages=len(state.get("messages", [])) - 1,
                has_summary=bool(state.get("summary")),
            )
        ],
    }
