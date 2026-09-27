"""Chat endpoints: one turn of the LangGraph pipeline, blocking or streamed (SSE).

Streaming protocol (`POST /sessions/{id}/chat/stream`, text/event-stream):

    event: agent   data: {agent, decision, reasoning, latency_ms, ...}   as each agent finishes
    event: token   data: {"text": "..."}                                 answer tokens as generated
    event: done    data: ChatResponse                                    final answer, sources, routing
    event: error   data: {"detail": "..."}                               if the turn failed

Streamed tokens are the raw model output; the `done` event carries the final,
normalized answer (e.g. unified citation markers) that clients should keep.
"""

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session, sessionmaker
from starlette.background import BackgroundTask

from app.api.deps import (
    acquire_session_turn,
    get_db,
    get_graph,
    get_owned_session,
    get_sessionmaker,
    session_turn_lock,
)
from app.api.schemas import ChatRequest, ChatResponse
from app.db.models import ChatSession
from app.graph.graph import SYNTHESIS, turn_input
from app.graph.state import turn_timings
from app.metrics import record_turn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["chat"])


def build_response(
    chat_session: ChatSession, state: dict[str, Any], latency_ms: float, trace_id: str | None = None
) -> ChatResponse:
    return ChatResponse(
        session_id=chat_session.id,
        turn=state.get("turn", 0),
        answer=state.get("answer", ""),
        sources=state.get("sources", []),
        intent=state.get("intent"),
        rewritten_query=state.get("rewritten_query") or None,
        sub_queries=state.get("sub_queries", []),
        retrieval_strategy=state.get("retrieval_strategy") if state.get("retrieved") else None,
        filters=state.get("filters", {}),
        memory_updates=state.get("memory_updates", []),
        latency_ms=round(latency_ms, 1),
        agent_latency_ms=turn_timings(state),
        trace_id=trace_id,
    )


@router.post(
    "/{session_id}/chat",
    response_model=ChatResponse,
    summary="Send a message and get the full answer",
    description="Runs one turn of the multi-agent pipeline (understanding, rewriting, routing, retrieval, "
    "synthesis, summarization, memory) with the session's history and the user's long-term memory.",
)
def chat(
    body: ChatRequest,
    chat_session: ChatSession = Depends(get_owned_session),
    db: Session = Depends(get_db),
    graph: CompiledStateGraph = Depends(get_graph),
) -> ChatResponse:
    inputs, config = turn_input(body.message, chat_session.user_id, chat_session.id)
    trace_id = str(config["run_id"])
    with session_turn_lock(chat_session.id):
        start = time.perf_counter()
        try:
            state = graph.invoke(inputs, config)
        except Exception as exc:
            logger.exception("Turn failed for session %s", chat_session.id)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Could not answer: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
    record_turn(db, chat_session, state, latency_ms, trace_id=trace_id)
    return build_response(chat_session, state, latency_ms, trace_id)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post(
    "/{session_id}/chat/stream",
    summary="Send a message and stream the answer (Server-Sent Events)",
    description="Same pipeline as `/chat`, streamed: `agent` events as each agent decides, `token` events "
    "with the answer as it is generated, then a `done` event with the full ChatResponse.",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def chat_stream(
    body: ChatRequest,
    chat_session: ChatSession = Depends(get_owned_session),
    factory: sessionmaker[Session] = Depends(get_sessionmaker),
    graph: CompiledStateGraph = Depends(get_graph),
) -> StreamingResponse:
    inputs, config = turn_input(body.message, chat_session.user_id, chat_session.id)
    session_id, trace_id = chat_session.id, str(config["run_id"])
    release = acquire_session_turn(session_id)  # 409 now, before the stream starts, if a turn is running

    def events() -> Iterator[str]:
        start = time.perf_counter()
        first_token_ms = None
        try:
            for mode, payload in graph.stream(inputs, config, stream_mode=["updates", "messages"]):
                if mode == "messages":
                    chunk, meta = payload
                    if meta.get("langgraph_node") == SYNTHESIS and isinstance(chunk, AIMessageChunk):
                        if text := str(chunk.text):
                            first_token_ms = first_token_ms or (time.perf_counter() - start) * 1000
                            yield _sse("token", {"text": text})
                else:
                    for update in payload.values():
                        for d in (update or {}).get("decisions", []):
                            yield _sse("agent", d)
            state = graph.get_state(config).values
            latency_ms = (time.perf_counter() - start) * 1000
            # The request's DB session is closed once streaming starts; use a fresh one.
            with factory() as db:
                owned = db.get(ChatSession, session_id)
                record_turn(db, owned, state, latency_ms, first_token_ms, trace_id=trace_id)
                yield _sse("done", build_response(owned, state, latency_ms, trace_id).model_dump(mode="json"))
        except Exception as exc:
            logger.exception("Streaming turn failed for session %s", session_id)
            yield _sse("error", {"detail": f"Could not answer: {exc}"})
        finally:
            release()

    # The background task also releases the lock if the client disconnects before streaming begins.
    return StreamingResponse(events(), media_type="text/event-stream", background=BackgroundTask(release),
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
