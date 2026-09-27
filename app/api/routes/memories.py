"""Long-term memory endpoints: what the assistant remembers about the caller, and forgetting it."""

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.graph.state import CompiledStateGraph

from app.api.deps import get_graph, get_user_id
from app.api.schemas import MemoryOut
from app.memory.long_term import delete_memory, user_namespace

router = APIRouter(prefix="/memories", tags=["memory"])


def _store(graph: CompiledStateGraph):
    if graph.store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Long-term memory store is not configured")
    return graph.store


@router.get("", response_model=list[MemoryOut], summary="List the caller's long-term memories")
def list_memories(user_id: str = Depends(get_user_id), graph: CompiledStateGraph = Depends(get_graph)) -> list[MemoryOut]:
    items = _store(graph).search(user_namespace(user_id), limit=100)
    return [MemoryOut(key=item.key, **item.value) for item in items]


@router.delete("/{key}", status_code=status.HTTP_204_NO_CONTENT, summary="Forget one memory")
def forget_memory(key: str, user_id: str = Depends(get_user_id), graph: CompiledStateGraph = Depends(get_graph)) -> None:
    store = _store(graph)
    if store.get(user_namespace(user_id), key) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Memory not found")
    delete_memory(store, user_id, key)
