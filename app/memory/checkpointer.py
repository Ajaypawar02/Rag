"""PostgreSQL connection pool and LangGraph checkpointer (short-term, thread-scoped memory).

The checkpointer saves the full graph state after every node, keyed by
`thread_id` (= session_id). Resuming a session loads its latest checkpoint,
so conversations survive API restarts, and sessions never see each other's
state.
"""

from functools import lru_cache

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings


@lru_cache(maxsize=1)
def get_pg_pool() -> ConnectionPool:
    """Connection pool shared by the checkpointer and the long-term store.

    LangGraph's Postgres classes require autocommit and dict rows.
    """
    return ConnectionPool(
        conninfo=get_settings().postgres_url,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )


@lru_cache(maxsize=1)
def get_checkpointer() -> PostgresSaver:
    saver = PostgresSaver(get_pg_pool())
    saver.setup()  # idempotent: creates/migrates checkpoint tables
    return saver
