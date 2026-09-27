"""Metadata extraction for documents and chunks.

Document-level: doc_type (file format), doc_category, product, version, title.
Chunk-level: section headers (set by the chunker) and code-block statistics.

All values are scalars because Chroma metadata only accepts str/int/float/bool;
`None` values are dropped before storage.
"""

import re
from datetime import datetime, timezone
from typing import Literal

from app.ingestion.loaders import FENCE_RE, LoadedDocument

Origin = Literal["seed", "upload"]

KNOWN_PRODUCTS = (
    "fastapi", "pydantic", "docker", "langgraph", "langchain", "redis",
    "kubernetes", "postgresql", "python", "chroma",
)

# Checked in order against "<filename> <title>"; first match wins.
_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    (r"migrat", "migration_guide"),
    (r"what'?s[ _-]?new|release[ _-]?notes|changelog", "release_notes"),
    (r"tutorial|first[ _-]?steps|getting[ _-]?started|quick[ _-]?start", "tutorial"),
    (r"reference|\bapi\b", "reference"),
    (r"how[ _-]?to|guide", "guide"),
)

_PRODUCT_VERSION_RE = re.compile(
    r"\b(?:" + "|".join(KNOWN_PRODUCTS) + r")\b[^\d\n]{0,20}?v?(\d{1,2}(?:\.\d{1,3}){0,2})\b", re.I
)
_FILENAME_VERSION_RE = re.compile(r"(?:^|_)v?(\d{1,2}(?:_\d{1,2}){0,2})(?=_|$)", re.I)
_FENCE_LANG_RE = re.compile(r"\A[ \t]*(?:`{3,}|~{3,})([\w+#.-]*)")


def detect_version(source: str, title: str) -> str | None:
    """Documented product version, from "<product> <version>" in the title, else the filename.

    Body text is deliberately not scanned: it mentions many unrelated versions
    ("added in version 0.95.0", "image nginx:1.14.2") that would mislabel the doc.
    """
    if m := _PRODUCT_VERSION_RE.search(title):
        return m.group(1)
    if m := _FILENAME_VERSION_RE.search(source.rsplit(".", 1)[0]):
        return m.group(1).replace("_", ".")
    return None


def detect_product(source: str, title: str) -> str | None:
    haystack = f"{source} {title}".lower()
    return next((p for p in KNOWN_PRODUCTS if p in haystack), None)


def detect_category(source: str, title: str) -> str:
    haystack = f"{source} {title}".lower()
    for pattern, category in _CATEGORY_RULES:
        if re.search(pattern, haystack):
            return category
    return "concept"


def build_document_metadata(doc: LoadedDocument, origin: Origin) -> dict:
    """Metadata shared by every chunk of a document."""
    return drop_none(
        {
            "source": doc.source,
            "title": doc.title,
            "doc_type": doc.doc_type,
            "doc_category": detect_category(doc.source, doc.title),
            "product": detect_product(doc.source, doc.title),
            "version": detect_version(doc.source, doc.title),
            "description": doc.description,
            "content_hash": doc.content_hash,
            "origin": origin,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )


def code_block_metadata(text: str) -> dict:
    """Code statistics for one chunk: presence, count and languages of fenced blocks."""
    blocks = [m.group(0) for m in FENCE_RE.finditer(text)]
    languages = sorted({lang for b in blocks if (lang := _FENCE_LANG_RE.match(b).group(1).lower())})
    return {
        # A long code block may be split across chunks, leaving an unclosed fence.
        "has_code": bool(blocks) or "```" in text,
        "code_block_count": len(blocks),
        "code_languages": ",".join(languages) or None,
    }


def drop_none(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if v is not None}
