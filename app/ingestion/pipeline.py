"""End-to-end ingestion: load -> metadata -> chunk -> embed -> store.

Shared by the seed script (origin="seed") and the upload endpoint
(origin="upload"). Re-ingesting a file with the same name replaces its chunks,
so ingestion is idempotent.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import get_settings
from app.ingestion.chunking import chunk_document
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, IngestionError, load_document
from app.ingestion.metadata import Origin, build_document_metadata
from app.memory.keyword_index import invalidate_keyword_index
from app.memory.vectorstore import add_chunks, delete_source, get_catalog

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    source: str
    title: str
    doc_type: str
    doc_category: str
    product: str | None
    version: str | None
    chunks: int
    chunks_with_code: int
    replaced_chunks: int
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def ingest_file(path: Path, origin: Origin = "upload") -> IngestResult:
    """Ingest one file into the vector store, replacing any previous version of it."""
    settings = get_settings()
    start = time.perf_counter()

    doc = load_document(path)
    doc_metadata = build_document_metadata(doc, origin)
    chunks = chunk_document(doc, doc_metadata, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise IngestionError(f"'{path.name}' produced no chunks.")

    replaced = delete_source(doc.source)
    add_chunks(chunks)
    invalidate_keyword_index()
    get_catalog.cache_clear()

    result = IngestResult(
        source=doc.source,
        title=doc.title,
        doc_type=doc.doc_type,
        doc_category=doc_metadata["doc_category"],
        product=doc_metadata.get("product"),
        version=doc_metadata.get("version"),
        chunks=len(chunks),
        chunks_with_code=sum(c.metadata["has_code"] for c in chunks),
        replaced_chunks=replaced,
        seconds=round(time.perf_counter() - start, 2),
    )
    logger.info("Ingested %s: %d chunks in %.2fs", doc.source, result.chunks, result.seconds)
    return result


def remove_document(source: str) -> int:
    """Delete a document's chunks from the knowledge base. Returns chunks removed."""
    removed = delete_source(source)
    invalidate_keyword_index()
    get_catalog.cache_clear()
    return removed


def ingest_directory(directory: Path, origin: Origin = "seed") -> tuple[list[IngestResult], dict[str, str]]:
    """Ingest every supported file in `directory`. Returns (results, {file: error})."""
    results, errors = [], {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            results.append(ingest_file(path, origin))
        except IngestionError as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            errors[path.name] = str(exc)
    return results, errors
