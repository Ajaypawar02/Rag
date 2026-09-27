"""Conversation Summarization Agent: bounds conversation memory once a thread grows long.

Runs after the answer when the thread exceeds SUMMARY_THRESHOLD messages.
Messages older than the last WINDOW_SIZE are removed from the checkpointed
state; with the summary_buffer strategy they are first folded into the
rolling summary (target length = COMPRESSION_RATIO x their text), with
sliding_window they are simply dropped. See app/memory/pruning.py.
"""

from langchain_core.messages import RemoveMessage

from app import llm
from app.config import get_settings
from app.graph import prompts
from app.graph.state import ConversationState
from app.memory.pruning import messages_to_prune, should_prune, summary_target_words, word_count
from app.observability.decision_log import decision, traced


def needs_summarization(state: ConversationState) -> bool:
    return should_prune(state.get("messages", []), get_settings().summary_threshold)


@traced("summarization")
def summarize_conversation(state: ConversationState) -> dict:
    s = get_settings()
    old = messages_to_prune(state.get("messages", []), s.window_size)
    if not old:
        return {"decisions": [decision("nothing to prune")]}
    removals = [RemoveMessage(id=m.id) for m in old]

    if s.memory_strategy == "sliding_window":
        return {
            "messages": removals,
            "decisions": [decision(f"dropped {len(old)} old messages", "sliding_window strategy: no summary kept.")],
        }

    existing = state.get("summary", "")
    source_text = prompts.render_history(old, existing)
    target = summary_target_words(source_text, s.compression_ratio)
    try:
        summary = llm.invoke_text(prompts.summarization_messages(existing, old, target), fast=True)
    except Exception as exc:
        # Keep the messages rather than lose them without a summary.
        return {"decisions": [decision("summarization skipped", f"Model error, messages kept: {exc}")]}

    before, after = word_count(source_text), word_count(summary)
    return {
        "summary": summary,
        "messages": removals,
        "decisions": [
            decision(
                f"folded {len(old)} messages into summary ({before} -> {after} words)",
                f"summary_buffer strategy, target compression {s.compression_ratio:.0%} (~{target} words).",
                pruned_messages=len(old),
                words_before=before,
                words_after=after,
                achieved_ratio=round(after / before, 2) if before else None,
            )
        ],
    }
