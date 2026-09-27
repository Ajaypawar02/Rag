"""Chroma vector store: client, LangChain wrapper and collection helpers.

In docker-compose Chroma runs as its own service (`CHROMA_MODE=http`); for
quick local runs without Docker it can be embedded on disk (`CHROMA_MODE=local`).
"""

from collections import defaultdict
from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.llm import get_embeddings

_ADD_BATCH_SIZE = 256


@lru_cache(maxsize=1)
def get_chroma_client() -> ClientAPI:
    s = get_settings()
    client_settings = ChromaSettings(anonymized_telemetry=False)
    if s.chroma_mode == "local":
        s.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(s.chroma_persist_dir), settings=client_settings)
    return chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port, settings=client_settings)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """LangChain Chroma wrapper used for semantic search."""
    return Chroma(
        client=get_chroma_client(),
        collection_name=get_settings().chroma_collection,
        embedding_function=get_embeddings(),
    )


def get_collection() -> Collection:
    """Raw Chroma collection, for metadata-only operations that need no embeddings."""
    return get_chroma_client().get_or_create_collection(get_settings().chroma_collection)


def add_chunks(chunks: list[Document]) -> None:
    store = get_vectorstore()
    for i in range(0, len(chunks), _ADD_BATCH_SIZE):
        batch = chunks[i : i + _ADD_BATCH_SIZE]
        store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])


def delete_source(source: str) -> int:
    """Delete all chunks of one document. Returns the number of chunks removed."""
    collection = get_collection()
    ids = collection.get(where={"source": source}, include=[])["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def get_all_chunks() -> list[Document]:
    """Every stored chunk (text + metadata); used to build the BM25 keyword index."""
    data = get_collection().get(include=["documents", "metadatas"])
    return [
        Document(id=id_, page_content=text or "", metadata=meta or {})
        for id_, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]


_DOCUMENT_FIELDS = ("title", "doc_type", "doc_category", "product", "version", "origin", "ingested_at")


def list_documents() -> list[dict]:
    """One summary row per ingested document, aggregated from chunk metadata."""
    metadatas = get_collection().get(include=["metadatas"])["metadatas"]
    docs: dict[str, dict] = {}
    counts: dict[str, int] = defaultdict(int)
    for meta in metadatas:
        source = meta["source"]
        counts[source] += 1
        docs.setdefault(source, {"source": source, **{f: meta.get(f) for f in _DOCUMENT_FIELDS}})
    return [{**docs[s], "chunks": counts[s]} for s in sorted(docs)]


@lru_cache(maxsize=1)
def get_catalog() -> tuple[dict, ...]:
    """Cached document list for agent prompts (what products/versions exist).

    Cleared by the ingestion pipeline whenever documents change.
    """
    return tuple(list_documents())


def reset_collection() -> None:
    """Drop and recreate the collection (used by `seed_docs --reset`)."""
    client = get_chroma_client()
    name = get_settings().chroma_collection
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    get_vectorstore.cache_clear()
    get_catalog.cache_clear()
