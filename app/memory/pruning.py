"""Conversation memory pruning.

Two strategies keep the prompt size bounded as a conversation grows
(`MEMORY_STRATEGY`):

- sliding_window: keep the last `WINDOW_SIZE` messages, drop older ones.
- summary_buffer: same window, but older messages are first folded into a
  rolling summary whose length is `COMPRESSION_RATIO` x the text it replaces.

Pruning triggers once the thread holds more than `SUMMARY_THRESHOLD` messages.
"""

from langchain_core.messages import AnyMessage, HumanMessage

MIN_SUMMARY_WORDS = 40
MAX_SUMMARY_WORDS = 400


def should_prune(messages: list[AnyMessage], threshold: int) -> bool:
    return len(messages) > threshold


def messages_to_prune(messages: list[AnyMessage], keep_last: int) -> list[AnyMessage]:
    """Messages older than the last `keep_last`, never splitting a user/assistant exchange.

    The kept window always starts at a human message, so the cut moves back
    (keeping slightly more) rather than orphaning an assistant reply.
    """
    if len(messages) <= keep_last:
        return []
    cutoff = len(messages) - keep_last
    while cutoff > 0 and not isinstance(messages[cutoff], HumanMessage):
        cutoff -= 1
    return messages[:cutoff]


def word_count(text: str) -> int:
    return len(text.split())


def summary_target_words(source_text: str, compression_ratio: float) -> int:
    """Target summary length: `compression_ratio` of the source, clamped to a sane range."""
    target = int(word_count(source_text) * compression_ratio)
    return max(MIN_SUMMARY_WORDS, min(MAX_SUMMARY_WORDS, target))
