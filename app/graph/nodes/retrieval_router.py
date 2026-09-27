"""Retrieval Router Agent: picks the search strategy and metadata filters for the query.

The model sees a catalog of the ingested documents (product, version, title)
so it can only target what exists; its filters are then validated against
that catalog, and anything invalid is dropped rather than causing an empty
search.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app import llm
from app.graph import prompts
from app.graph.state import ConversationState
from app.memory.vectorstore import get_catalog
from app.observability.decision_log import decision, traced


class RouterFilters(BaseModel):
    source: str | None = Field(default=None, description="Exact source file name of one document, or null.")
    product: str | None = Field(default=None, description="One product from the catalog, or null.")
    version: str | None = Field(default=None, description="A version listed for that product, or null.")
    has_code: bool | None = Field(default=None, description="True only if the user wants code/examples.")


class RetrievalPlan(BaseModel):
    strategy: Literal["hybrid", "semantic", "keyword"]
    filters: RouterFilters = Field(default_factory=RouterFilters)
    reasoning: str = Field(description="One sentence explaining the plan.")


def validate_filters(filters: RouterFilters, catalog: list[dict] | tuple[dict, ...]) -> tuple[dict, list[str]]:
    """Keep only filters that match the catalog. Returns (filters, notes about dropped ones)."""
    valid: dict = {}
    notes: list[str] = []
    products = {d.get("product") for d in catalog if d.get("product")}

    source = (filters.source or "").strip() or None
    if source and source in {d["source"] for d in catalog}:
        valid["source"] = source  # one document: product/version would be redundant
        if filters.has_code:
            valid["has_code"] = True
        return valid, notes
    if source:
        notes.append(f"dropped unknown source '{filters.source}'")

    product = (filters.product or "").strip().lower() or None
    if product and product in products:
        valid["product"] = product
    elif product:
        notes.append(f"dropped unknown product '{filters.product}'")

    if filters.version:
        versions = {str(d.get("version")) for d in catalog if d.get("version") and d.get("product") == valid.get("product")}
        if valid.get("product") and filters.version in versions:
            valid["version"] = filters.version
        else:
            notes.append(f"dropped version '{filters.version}' (not in catalog for this product)")

    if filters.has_code:
        valid["has_code"] = True
    return valid, notes


@traced("retrieval_router")
def route_retrieval(state: ConversationState) -> dict:
    query = state.get("rewritten_query") or state.get("query", "")
    sub_queries = state.get("sub_queries", [])
    catalog = get_catalog()

    try:
        plan = llm.invoke_structured(RetrievalPlan, prompts.router_messages(query, sub_queries, catalog), fast=True)
        fallback = False
    except Exception as exc:
        plan = RetrievalPlan(strategy="hybrid", reasoning=f"Fallback after model error: {exc}")
        fallback = True

    filters, notes = validate_filters(plan.filters, catalog)
    strategy = "multi_query" if len(sub_queries) > 1 else plan.strategy
    return {
        "retrieval_strategy": strategy,
        "filters": filters,
        "decisions": [
            decision(
                f"strategy={strategy}, filters={filters or 'none'}",
                plan.reasoning + (f" ({'; '.join(notes)})" if notes else ""),
                proposed_strategy=plan.strategy,
                proposed_filters=plan.filters.model_dump(exclude_none=True),
                fallback=fallback,
            )
        ],
    }
