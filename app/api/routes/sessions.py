"""Session endpoints: create (returns session_id), list, get, delete, history."""

import uuid

from fastapi import APIRouter, Depends, status
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_graph, get_owned_session, get_user_id
from app.api.schemas import HistoryOut, MessageOut, SessionCreate, SessionOut, TranscriptTurnOut
from app.db.models import ChatSession, TurnRecord

router = APIRouter(prefix="/sessions", tags=["sessions"])


def thread_config(chat_session: ChatSession) -> dict:
    return {"configurable": {"thread_id": chat_session.id, "user_id": chat_session.user_id}}


@router.post(
    "",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new conversation session",
    description="Creates a session owned by the caller and returns its `session_id` for use in chat requests.",
)
def create_session(
    body: SessionCreate | None = None,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
) -> ChatSession:
    chat_session = ChatSession(id=uuid.uuid4().hex, user_id=user_id, title=body.title if body else None)
    db.add(chat_session)
    db.commit()
    return chat_session


@router.get("", response_model=list[SessionOut], summary="List the caller's sessions (most recent first)")
def list_sessions(user_id: str = Depends(get_user_id), db: Session = Depends(get_db)) -> list[ChatSession]:
    query = select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.updated_at.desc())
    return list(db.scalars(query))


@router.get("/{session_id}", response_model=SessionOut, summary="Get one session")
def get_session(chat_session: ChatSession = Depends(get_owned_session)) -> ChatSession:
    return chat_session


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session",
    description="Deletes the session, its turn records and its conversation checkpoints. "
    "The user's long-term memories are kept.",
)
def delete_session(
    chat_session: ChatSession = Depends(get_owned_session),
    db: Session = Depends(get_db),
    graph: CompiledStateGraph = Depends(get_graph),
) -> None:
    if graph.checkpointer is not None:
        graph.checkpointer.delete_thread(chat_session.id)
    db.execute(delete(TurnRecord).where(TurnRecord.session_id == chat_session.id))
    db.delete(chat_session)  # same DB session as get_owned_session (dependencies are cached per request)
    db.commit()


@router.get(
    "/{session_id}/transcript",
    response_model=list[TranscriptTurnOut],
    summary="Full conversation transcript (for display)",
    description="Every turn's question, answer and sources, oldest first. Unlike `/history`, this is never "
    "pruned by summarization - it is what a chat UI shows.",
)
def get_transcript(
    chat_session: ChatSession = Depends(get_owned_session),
    db: Session = Depends(get_db),
) -> list[TurnRecord]:
    query = select(TurnRecord).where(TurnRecord.session_id == chat_session.id).order_by(TurnRecord.turn)
    return list(db.scalars(query))


@router.get(
    "/{session_id}/history",
    response_model=HistoryOut,
    summary="Conversation memory of a session (model's view)",
    description="The messages the model currently sees plus the rolling summary of older, pruned messages.",
)
def get_history(
    chat_session: ChatSession = Depends(get_owned_session),
    graph: CompiledStateGraph = Depends(get_graph),
) -> HistoryOut:
    values = graph.get_state(thread_config(chat_session)).values
    messages = [
        MessageOut(role="user" if isinstance(m, HumanMessage) else "assistant", content=str(m.text))
        for m in values.get("messages", [])
        if isinstance(m, (HumanMessage, AIMessage))
    ]
    return HistoryOut(session_id=chat_session.id, summary=values.get("summary", ""), messages=messages)
