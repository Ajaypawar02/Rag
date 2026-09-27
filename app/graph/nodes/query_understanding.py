"""Query Understanding Agent: classifies intent and whether the query needs conversation history."""

from pydantic import BaseModel, Field

from app import llm
from app.graph import prompts
from app.graph.state import ConversationState, Intent
from app.memory.vectorstore import get_catalog
from app.observability.decision_log import decision, traced


class QueryAnalysis(BaseModel):
    intent: Intent
    needs_history: bool = Field(description="True if the message cannot be understood without earlier turns.")
    reasoning: str = Field(description="One sentence explaining the classification.")


@traced("query_understanding")
def understand_query(state: ConversationState) -> dict:
    query = state.get("query", "")
    history = prompts.history_before_latest(state.get("messages", []))
    has_history = bool(history) or bool(state.get("summary"))

    try:
        analysis = llm.invoke_structured(
            QueryAnalysis,
            prompts.understanding_messages(query, history, state.get("summary", ""), prompts.catalog_topics(get_catalog())),
            fast=True,
        )
        fallback = False
    except Exception as exc:
        # Safe default: treat as a question and rewrite it if there is any history to draw on.
        analysis = QueryAnalysis(
            intent="follow_up" if has_history else "new_question",
            needs_history=has_history,
            reasoning=f"Fallback after model error: {exc}",
        )
        fallback = True

    intent, needs_history = analysis.intent, analysis.needs_history
    if not has_history:  # nothing to resolve against on the first turn
        needs_history = False
        intent = "new_question" if intent == "follow_up" else intent
    elif intent == "follow_up":
        needs_history = True

    return {
        "intent": intent,
        "needs_history": needs_history,
        "decisions": [
            decision(f"intent={intent}, needs_history={needs_history}", analysis.reasoning, fallback=fallback)
        ],
    }
