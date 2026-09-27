"""Agent decision logging.

`@traced("agent_name")` wraps a graph node so that every decision it reports
is stamped with the agent name, turn number and latency, written to the
application log, and appended to the state's per-turn `decisions` list — the
data behind the explainability dashboard and per-agent latency metrics.

A node reports decisions by returning
    {"decisions": [{"decision": "...", "reasoning": "...", "details": {...}}]}
and gets a minimal "completed" entry if it reports none, so every executed
node appears in the log with its timing.
"""

import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("agent_decisions")


def decision(summary: str, reasoning: str = "", **details: Any) -> dict:
    """Build an (unstamped) decision entry for a node's return value."""
    return {"decision": summary, "reasoning": reasoning, "details": details}


def traced(agent: str) -> Callable:
    def wrap(node: Callable[..., dict]) -> Callable[..., dict]:
        @functools.wraps(node)  # keeps the signature so LangGraph still injects `store`, `config`, ...
        def inner(state: dict, *args: Any, **kwargs: Any) -> dict:
            start = time.perf_counter()
            update = node(state, *args, **kwargs) or {}
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            turn = update.get("turn", state.get("turn", 0))  # load_context sets the new turn itself

            entries = update.get("decisions") or [decision("completed")]
            stamped = [{**entry, "agent": agent, "turn": turn, "latency_ms": latency_ms} for entry in entries]
            for entry in stamped:
                logger.info("turn=%s %s: %s (%.0f ms) %s", turn, agent, entry["decision"], latency_ms,
                            entry.get("reasoning", ""))
            return {**update, "decisions": stamped}

        return inner

    return wrap
