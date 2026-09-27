"""System metrics: latency, agent timings, routing mix and memory overhead."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.schemas import MetricsOut
from app.metrics import aggregate_turns, memory_overhead

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_model=MetricsOut,
    summary="Performance metrics",
    description="Aggregated over all answered turns: end-to-end and time-to-first-token latency, mean latency "
    "per agent, intent/strategy mix, citation rate, and PostgreSQL memory overhead (checkpoints + store).",
)
def get_metrics(db: Session = Depends(get_db)) -> MetricsOut:
    return MetricsOut(**aggregate_turns(db), memory=memory_overhead())
