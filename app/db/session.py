"""SQLAlchemy engine and session factory for the application tables.

Uses the same PostgreSQL database as LangGraph (via the psycopg 3 driver).
Tests swap in SQLite through the `get_sessionmaker` dependency.
"""

from functools import lru_cache

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base


def sqlalchemy_url(postgres_url: str) -> str:
    """postgresql://... -> postgresql+psycopg://... (psycopg 3, already installed for LangGraph)."""
    for prefix in ("postgresql://", "postgres://"):
        if postgres_url.startswith(prefix):
            return "postgresql+psycopg://" + postgres_url[len(prefix):]
    return postgres_url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(sqlalchemy_url(get_settings().postgres_url), pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def init_db(engine: Engine | None = None) -> None:
    """Create application tables if missing and add any new nullable columns (idempotent).

    A minimal stand-in for migrations: enough for additive schema changes in
    this prototype; a production system would use Alembic.
    """
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing and column.nullable:
                    col_type = column.type.compile(dialect=engine.dialect)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {column.name} {col_type}'))
