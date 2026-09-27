"""Query Decomposition Agent: splits a multi-part question into standalone sub-queries.

The sub-queries are searched separately and fused (the `multi_query`
retrieval strategy), so each part of the question gets its own evidence.
"""

from pydantic import BaseModel, Field

from app import llm
from app.graph import prompts
from app.graph.state import ConversationState
from app.observability.decision_log import decision, traced

MAX_SUB_QUERIES = 4


class Decomposition(BaseModel):
    sub_queries: list[str] = Field(description="2-4 standalone sub-questions, or the single original question.")
    reasoning: str = Field(description="One sentence on how the question was split.")


@traced("query_decomposition")
def decompose_query(state: ConversationState) -> dict:
    query = state.get("rewritten_query") or state.get("query", "")
    try:
        result = llm.invoke_structured(
            Decomposition, prompts.decomposition_messages(query, state.get("user_memories", [])), fast=True
        )
        sub_queries = [q.strip() for q in result.sub_queries if q.strip()][:MAX_SUB_QUERIES]
        reasoning, fallback = result.reasoning, False
    except Exception as exc:
        sub_queries, reasoning, fallback = [], f"Fallback after model error, searching the full query: {exc}", True

    if len(sub_queries) < 2:  # not actually decomposable: search the query as a whole
        sub_queries = []
    return {
        "sub_queries": sub_queries,
        "decisions": [
            decision(f"{len(sub_queries) or 1} search quer{'ies' if sub_queries else 'y'}", reasoning,
                     sub_queries=sub_queries, fallback=fallback)
        ],
    }
