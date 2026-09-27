"""Query Rewriting Agent: turns a context-dependent message into a standalone search query."""

from pydantic import BaseModel, Field

from app import llm
from app.graph import prompts
from app.graph.state import ConversationState
from app.observability.decision_log import decision, traced


class RewrittenQuery(BaseModel):
    standalone_query: str = Field(description="The query rewritten to be understandable without the conversation.")
    reasoning: str = Field(description="One sentence on what was resolved or added.")


@traced("query_rewriting")
def rewrite_query(state: ConversationState) -> dict:
    query = state.get("query", "")
    history = prompts.history_before_latest(state.get("messages", []))
    try:
        result = llm.invoke_structured(
            RewrittenQuery,
            prompts.rewriting_messages(query, history, state.get("summary", ""), state.get("user_memories", [])),
            fast=True,
        )
        rewritten, reasoning, fallback = result.standalone_query.strip() or query, result.reasoning, False
    except Exception as exc:
        rewritten, reasoning, fallback = query, f"Fallback after model error, using the original query: {exc}", True

    return {
        "rewritten_query": rewritten,
        "decisions": [decision(f"'{query}' -> '{rewritten}'", reasoning, original=query, fallback=fallback)],
    }
