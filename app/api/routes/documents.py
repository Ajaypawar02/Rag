"""Document endpoints: upload + process, list, delete."""

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.api.schemas import DeleteOut, DocumentOut, IngestOut
from app.config import get_settings
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, IngestionError, UnsupportedFileTypeError
from app.ingestion.pipeline import ingest_file, remove_document
from app.memory.vectorstore import list_documents

router = APIRouter(prefix="/documents", tags=["documents"])

_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    """Strip directories and unusual characters so uploads cannot escape the upload dir."""
    cleaned = _UNSAFE_CHARS_RE.sub("_", Path(name).name).strip("._")
    return cleaned or "upload"


@router.post(
    "/upload",
    response_model=IngestOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a document",
    description=(
        "Upload a PDF, Markdown or HTML file. It is parsed, enriched with metadata (type, version, "
        "sections, code blocks), chunked, embedded and indexed for search. Uploading a file with an "
        "existing name replaces the previous version."
    ),
)
def upload_document(file: UploadFile) -> IngestOut:
    settings = get_settings()
    filename = safe_filename(file.filename or "")
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    content = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {settings.max_upload_mb} MB")
    if not content:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "File is empty")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / filename
    path.write_bytes(content)
    try:
        result = ingest_file(path, origin="upload")
    except UnsupportedFileTypeError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except IngestionError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return IngestOut(**result.to_dict())


@router.get("", response_model=list[DocumentOut], summary="List ingested documents")
def get_documents() -> list[dict]:
    return list_documents()


@router.delete("/{source}", response_model=DeleteOut, summary="Remove a document from the knowledge base")
def delete_document(source: str) -> DeleteOut:
    removed = remove_document(source)
    if not removed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    (get_settings().upload_dir / safe_filename(source)).unlink(missing_ok=True)
    return DeleteOut(deleted=source, chunks_removed=removed)
