"""Long-term, cross-session user memory (LangGraph Store on PostgreSQL + pgvector).

Facts are stored per user under the namespace ("memories", user_id), so every
session of the same user can read them while other users never can. Each
item is embedded (same model as the documents), which lets the load-context
node fetch only the memories relevant to the current question.
"""

import uuid
from datetime import datetime, timezone
from functools import lru_cache

from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore

from app.llm import get_embeddings
from app.memory.checkpointer import get_pg_pool

MEMORY_NAMESPACE = "memories"


def user_namespace(user_id: str) -> tuple[str, str]:
    return (MEMORY_NAMESPACE, user_id)


@lru_cache(maxsize=1)
def get_store() -> PostgresStore:
    embeddings = get_embeddings()
    dims = len(embeddings.embed_query("dimension probe"))
    store = PostgresStore(get_pg_pool(), index={"dims": dims, "embed": embeddings, "fields": ["text"]})
    store.setup()  # idempotent: creates store + vector tables
    return store


def search_memories(store: BaseStore, user_id: str, query: str = "", limit: int = 5) -> list[dict]:
    """Memories most relevant to `query` (semantic), or most recent when no query."""
    items = store.search(user_namespace(user_id), query=query or None, limit=limit)
    return [{"key": item.key, "score": getattr(item, "score", None), **item.value} for item in items]


def save_memory(store: BaseStore, user_id: str, text: str, category: str) -> str:
    key = uuid.uuid4().hex
    store.put(
        user_namespace(user_id),
        key,
        {"text": text, "category": category, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    )
    return key


def delete_memory(store: BaseStore, user_id: str, key: str) -> None:
    store.delete(user_namespace(user_id), key)
