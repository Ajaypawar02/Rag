"""Explainability: the per-turn agent decision trace of a session."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_owned_session
from app.api.schemas import TurnTraceOut
from app.db.models import ChatSession, TurnRecord

router = APIRouter(prefix="/sessions", tags=["explainability"])


@router.get(
    "/{session_id}/decisions",
    response_model=list[TurnTraceOut],
    summary="Agent decision trace per turn",
    description="For each turn: what every agent decided (intent, rewrite, strategy, filters, retrieval, "
    "memory), why, and how long it took. Newest turn first.",
)
def get_decisions(
    turn: int | None = Query(default=None, ge=1, description="Only this turn"),
    chat_session: ChatSession = Depends(get_owned_session),
    db: Session = Depends(get_db),
) -> list[TurnRecord]:
    query = select(TurnRecord).where(TurnRecord.session_id == chat_session.id).order_by(TurnRecord.turn.desc())
    if turn is not None:
        query = query.where(TurnRecord.turn == turn)
    return list(db.scalars(query))
