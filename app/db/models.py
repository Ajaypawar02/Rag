"""SQLAlchemy models for application data kept next to LangGraph's own tables.

- ChatSession: which user owns which conversation (session_id == LangGraph thread_id).
  LangGraph's checkpoint tables only know thread ids; this table adds ownership,
  titles and listing per user, which is what enforces session isolation.
- TurnRecord: one row per answered message - latency, routing outcome and the
  full agent decision log - feeding /metrics and the explainability dashboard.
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str | None] = mapped_column(String(200))
    turns: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TurnRecord(Base):
    __tablename__ = "turn_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    turn: Mapped[int] = mapped_column(Integer)
    question: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str | None] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(32))
    retrieval_strategy: Mapped[str | None] = mapped_column(String(32))
    retrieved_chunks: Mapped[int] = mapped_column(Integer, default=0)
    cited_sources: Mapped[int] = mapped_column(Integer, default=0)
    answer_chars: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float)
    first_token_ms: Mapped[float | None] = mapped_column(Float)  # streaming only
    agent_latency_ms: Mapped[dict] = mapped_column(JSON, default=dict)
    decisions: Mapped[list] = mapped_column(JSON, default=list)
    trace_id: Mapped[str | None] = mapped_column(String(36))  # LangSmith root run id of the turn
    # Display transcript: kept here in full even after summarization prunes the model's context.
    answer: Mapped[str | None] = mapped_column(Text)
    sources: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
