"""Context Synthesis Agent: writes the answer from retrieved docs, conversation and user memories.

Uses the main (larger) model. Excerpts are numbered in the prompt and the
answer cites them as [n]; the cited numbers are mapped back to their source
documents for the response. Chitchat and out-of-scope messages get a short
reply without documentation context.
"""

import re

from langchain_core.messages import AIMessage

from app import llm
from app.graph import prompts
from app.graph.state import ConversationState
from app.memory.vectorstore import get_catalog
from app.observability.decision_log import decision, traced

_CITATION_RE = re.compile(r"\[(\d+)\]")
_FULLWIDTH_CITATION_RE = re.compile(r"【(\d+)(?:†[^】]*)?】")  # 【5】 / 【5†source】 used by some models
_REPEATED_CITATION_RE = re.compile(r"(\[\d+\])(?:\s*\1)+")

ERROR_ANSWER = "Sorry, I couldn't generate an answer right now. Please try again."


def normalize_citations(answer: str) -> str:
    """Unify citation markers to [n] and collapse immediate repeats ("[5]【5】" -> "[5]")."""
    answer = _FULLWIDTH_CITATION_RE.sub(r"[\1]", answer)
    return _REPEATED_CITATION_RE.sub(r"\1", answer)


def cited_sources(answer: str, retrieved: list[dict]) -> list[dict]:
    """Sources for the [n] markers used in the answer, in order of first citation."""
    sources, seen = [], set()
    for match in _CITATION_RE.finditer(answer):
        n = int(match.group(1))
        if n in seen or not 1 <= n <= len(retrieved):
            continue
        seen.add(n)
        chunk = retrieved[n - 1]
        source = {"index": n, **{k: chunk[k] for k in ("source", "title", "section", "page") if k in chunk}}
        sources.append(source)
    return sources


@traced("context_synthesis")
def synthesize_answer(state: ConversationState) -> dict:
    intent = state.get("intent", "new_question")
    retrieved = state.get("retrieved", [])
    messages = prompts.synthesis_messages(
        intent=intent,
        history=prompts.history_before_latest(state.get("messages", [])),
        summary=state.get("summary", ""),
        query=state.get("query", ""),
        retrieved=retrieved,
        memories=state.get("user_memories", []),
        products=prompts.catalog_topics(get_catalog()),
        filter_relaxed=state.get("retrieval_info", {}).get("filter_relaxed", False),
    )
    try:
        answer, error = normalize_citations(llm.invoke_text(messages, fast=False)), None
    except Exception as exc:
        answer, error = ERROR_ANSWER, str(exc)

    sources = cited_sources(answer, retrieved)
    reasoning = (
        f"Model error: {error}" if error
        else f"Grounded in {len(retrieved)} excerpts; cited {len(sources)}." if retrieved
        else f"No documentation context used (intent={intent})."
    )
    return {
        "answer": answer,
        "sources": sources,
        "messages": [AIMessage(answer)],
        "decisions": [
            decision(f"answered ({len(answer)} chars, {len(sources)} citations)", reasoning,
                     cited=[s["index"] for s in sources], error=error)
        ],
    }
