"""Metrics: record every answered turn and aggregate them for GET /metrics.

- Latency: end-to-end per turn, time to first streamed token, and per agent
  (from the decision log).
- Behaviour: intent and retrieval-strategy mix, chunks retrieved, citation rate.
- Memory overhead: bytes held in LangGraph's checkpoint and store tables.

Context preservation accuracy needs labelled conversations, so it is measured
offline by scripts/evaluate.py rather than here.
"""

import statistics
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatSession, TurnRecord, utcnow
from app.graph.state import turn_timings


def record_turn(
    db: Session,
    chat_session: ChatSession,
    state: dict[str, Any],
    latency_ms: float,
    first_token_ms: float | None = None,
    trace_id: str | None = None,
) -> TurnRecord:
    """Persist one answered turn and update the session's counters."""
    record = TurnRecord(
        session_id=chat_session.id,
        user_id=chat_session.user_id,
        turn=state.get("turn", 0),
        question=state.get("query", ""),
        rewritten_query=state.get("rewritten_query") or None,
        intent=state.get("intent"),
        retrieval_strategy=state.get("retrieval_strategy") if state.get("retrieved") else "none",
        retrieved_chunks=len(state.get("retrieved", [])),
        cited_sources=len(state.get("sources", [])),
        answer_chars=len(state.get("answer", "")),
        latency_ms=round(latency_ms, 1),
        first_token_ms=round(first_token_ms, 1) if first_token_ms is not None else None,
        agent_latency_ms=turn_timings(state),
        decisions=list(state.get("decisions", [])),
        trace_id=trace_id,
        answer=state.get("answer", ""),
        sources=list(state.get("sources", [])),
    )
    chat_session.turns = record.turn
    chat_session.updated_at = utcnow()
    if not chat_session.title:
        chat_session.title = record.question[:80]
    db.add(record)
    db.commit()
    return record


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "mean": None}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"p50": round(statistics.median(ordered), 1), "p95": round(p95, 1),
            "mean": round(statistics.fmean(ordered), 1)}


def aggregate_turns(db: Session) -> dict[str, Any]:
    records = db.scalars(select(TurnRecord)).all()
    agent_values: dict[str, list[float]] = defaultdict(list)
    for r in records:
        for agent, ms in (r.agent_latency_ms or {}).items():
            agent_values[agent].append(ms)

    doc_backed = [r for r in records if r.retrieved_chunks > 0]
    return {
        "turns": len(records),
        "latency_ms": _percentiles([r.latency_ms for r in records]),
        "first_token_ms": _percentiles([r.first_token_ms for r in records if r.first_token_ms is not None]),
        "agent_latency_ms": {a: round(statistics.fmean(v), 1) for a, v in sorted(agent_values.items())},
        "intents": dict(Counter(r.intent or "unknown" for r in records)),
        "retrieval_strategies": dict(Counter(r.retrieval_strategy or "none" for r in records)),
        "avg_retrieved_chunks": round(statistics.fmean(r.retrieved_chunks for r in doc_backed), 2) if doc_backed else None,
        "citation_rate": round(sum(r.cited_sources > 0 for r in doc_backed) / len(doc_backed), 3) if doc_backed else None,
    }


_MEMORY_SQL = {
    "threads": "select count(distinct thread_id) as n from checkpoints",
    "checkpoints": "select count(*) as n from checkpoints",
    "checkpoint_bytes": "select coalesce(sum(pg_column_size(checkpoint) + pg_column_size(metadata)), 0) as n from checkpoints",
    "blob_bytes": "select coalesce(sum(octet_length(blob)), 0) as n from checkpoint_blobs",
    "write_bytes": "select coalesce(sum(octet_length(blob)), 0) as n from checkpoint_writes",
    "long_term_memories": "select count(*) as n from store",
    "store_bytes": "select pg_total_relation_size('store') + pg_total_relation_size('store_vectors') as n",
}


def memory_overhead() -> dict[str, Any]:
    """Size of conversation state (checkpoints) and long-term memory (store) in PostgreSQL."""
    from app.memory.checkpointer import get_pg_pool

    try:
        with get_pg_pool().connection() as conn:
            values = {name: int(conn.execute(sql).fetchone()["n"]) for name, sql in _MEMORY_SQL.items()}
    except Exception as exc:  # metrics must not fail because a table is missing or DB is down
        return {"error": str(exc)}

    state_bytes = values["checkpoint_bytes"] + values["blob_bytes"] + values["write_bytes"]
    threads = values["threads"] or 1
    return {
        **values,
        "state_bytes_total": state_bytes,
        "state_kb_per_session": round(state_bytes / threads / 1024, 1),
        "checkpoints_per_session": round(values["checkpoints"] / threads, 1),
    }
