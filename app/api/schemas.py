"""Request and response models for the API (they also drive the OpenAPI/Swagger docs)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --- Documents ---------------------------------------------------------------

class IngestOut(BaseModel):
    source: str = Field(examples=["fastapi_dependencies.md"])
    title: str
    doc_type: str = Field(examples=["markdown"])
    doc_category: str = Field(examples=["tutorial"])
    product: str | None = Field(default=None, examples=["fastapi"])
    version: str | None = Field(default=None, examples=["2"])
    chunks: int
    chunks_with_code: int
    replaced_chunks: int = Field(description="Chunks of a previous version of this file that were replaced.")
    seconds: float


class DocumentOut(BaseModel):
    source: str
    title: str | None = None
    doc_type: str | None = None
    doc_category: str | None = None
    product: str | None = None
    version: str | None = None
    origin: str | None = Field(default=None, description="'seed' (sample docs) or 'upload'.")
    ingested_at: str | None = None
    chunks: int


class DeleteOut(BaseModel):
    deleted: str
    chunks_removed: int = 0


# --- Sessions ----------------------------------------------------------------

class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200, description="Optional; defaults to the first message.")


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str = Field(validation_alias="id")
    user_id: str
    title: str | None
    turns: int
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    role: str = Field(examples=["user", "assistant"])
    content: str


class HistoryOut(BaseModel):
    session_id: str
    summary: str = Field(description="Rolling summary of older messages that were pruned from the history.")
    messages: list[MessageOut]


# --- Chat --------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, examples=["How do I declare a dependency in FastAPI?"])


class SourceOut(BaseModel):
    index: int = Field(description="The [n] marker used in the answer.")
    source: str | None = None
    title: str | None = None
    section: str | None = None
    page: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    turn: int
    answer: str
    sources: list[SourceOut]
    intent: str | None = None
    rewritten_query: str | None = Field(default=None, description="Standalone query used for retrieval, if rewritten.")
    sub_queries: list[str] = []
    retrieval_strategy: str | None = None
    filters: dict[str, Any] = {}
    memory_updates: list[str] = Field(default=[], description="Facts stored in long-term memory this turn.")
    latency_ms: float
    agent_latency_ms: dict[str, float]
    trace_id: str | None = Field(default=None, description="LangSmith trace (root run) id of this turn.")


class TranscriptTurnOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    turn: int
    question: str
    answer: str | None
    sources: list[SourceOut] | None = None
    latency_ms: float
    trace_id: str | None = None
    created_at: datetime


# --- Decisions / explainability ---------------------------------------------

class DecisionOut(BaseModel):
    turn: int
    agent: str
    decision: str
    reasoning: str = ""
    details: dict[str, Any] = {}
    latency_ms: float


class TurnTraceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    turn: int
    question: str
    rewritten_query: str | None
    intent: str | None
    retrieval_strategy: str | None
    latency_ms: float
    first_token_ms: float | None
    decisions: list[DecisionOut]
    trace_id: str | None = None
    created_at: datetime


# --- Memory ------------------------------------------------------------------

class MemoryOut(BaseModel):
    key: str
    text: str
    category: str | None = None
    created_at: str | None = None


# --- Health / metrics --------------------------------------------------------

class HealthOut(BaseModel):
    status: str = Field(examples=["ok", "degraded"])
    components: dict[str, str]


class MetricsOut(BaseModel):
    turns: int
    latency_ms: dict[str, float | None] = Field(description="p50 / p95 / mean end-to-end turn latency.")
    first_token_ms: dict[str, float | None] = Field(description="Time to first streamed token (streaming turns).")
    agent_latency_ms: dict[str, float] = Field(description="Mean latency per agent.")
    intents: dict[str, int]
    retrieval_strategies: dict[str, int]
    avg_retrieved_chunks: float | None
    citation_rate: float | None = Field(description="Share of document-backed answers that cite a source.")
    memory: dict[str, Any] = Field(description="Memory overhead: checkpoint/store sizes in PostgreSQL.")
