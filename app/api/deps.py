"""Shared FastAPI dependencies: user identity, DB sessions, the graph, session ownership.

Identity is a plain `X-User-Id` header (no authentication - out of scope for
this prototype); it is what separates users' sessions and memories. Every
session-scoped route loads the session through `get_owned_session`, which
returns 404 for sessions that belong to someone else.
"""

import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from fastapi import Depends, Header, HTTPException, Path, status
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import ChatSession
from app.db.session import get_session_factory
from app.graph.graph import get_graph as _get_graph

_USER_ID_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


def get_sessionmaker() -> sessionmaker[Session]:
    return get_session_factory()


def get_db(factory: sessionmaker[Session] = Depends(get_sessionmaker)) -> Iterator[Session]:
    with factory() as db:
        yield db


def get_graph() -> CompiledStateGraph:
    return _get_graph()


def get_user_id(x_user_id: str = Header(..., description="Caller's user id; isolates sessions and memories.")) -> str:
    if not _USER_ID_RE.match(x_user_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "X-User-Id must be 1-64 chars of letters, digits, _ . @ -")
    return x_user_id


def get_owned_session(
    session_id: str = Path(..., max_length=64),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
) -> ChatSession:
    chat_session = db.get(ChatSession, session_id)
    if chat_session is None or chat_session.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return chat_session


# One turn at a time per session: two concurrent turns on one thread would race on its checkpoint.
_session_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def acquire_session_turn(session_id: str) -> Callable[[], None]:
    """Claim the session for one turn; returns an idempotent release function. 409 if busy."""
    with _locks_guard:
        lock = _session_locks.setdefault(session_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise HTTPException(status.HTTP_409_CONFLICT, "A message is already being processed for this session")
    released = threading.Event()

    def release() -> None:
        if not released.is_set():
            released.set()
            lock.release()

    return release


@contextmanager
def session_turn_lock(session_id: str) -> Iterator[None]:
    release = acquire_session_turn(session_id)
    try:
        yield
    finally:
        release()
