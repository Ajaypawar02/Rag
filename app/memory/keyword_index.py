"""BM25 keyword index over all stored chunks.

Complements semantic search for exact terms that embeddings handle poorly:
identifiers (`Depends`, `model_validate`), CLI flags, error messages, versions.
The retrieval layer fuses its results with Chroma's for hybrid search.

The index is built lazily from Chroma and rebuilt after every ingestion
(`invalidate_keyword_index`); rebuilding is cheap at documentation scale.
"""

import re
import threading

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.memory.vectorstore import get_all_chunks

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it of on or should that the this "
    "to what when where which who why with you your".split()
)


# Light suffix stripping so word forms match ("suppression"/"suppress", "predictions"/"prediction").
# Deliberately conservative: identifiers with underscores or digits are left untouched.
_SUFFIXES = ("ations", "ation", "ions", "ion", "ings", "ing", "ies", "ed", "es", "s")
_MIN_STEM = 4


def stem(token: str) -> str:
    if "_" in token or any(ch.isdigit() for ch in token):
        return token
    for suffix in _SUFFIXES:
        if suffix == "s" and token.endswith("ss"):  # "suppress", "class" are not plurals
            continue
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            base = token[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def matches_filter(metadata: dict, where: dict | None) -> bool:
    """Evaluate a Chroma-style `where` filter against one metadata dict.

    Supports `{"field": value}`, `{"field": {"$eq"|"$ne"|"$in"|"$nin": ...}}`,
    `{"$and": [...]}` and `{"$or": [...]}` — the subset the retrieval router emits.
    """
    if not where:
        return True
    for key, cond in where.items():
        if key == "$and":
            if not all(matches_filter(metadata, c) for c in cond):
                return False
        elif key == "$or":
            if not any(matches_filter(metadata, c) for c in cond):
                return False
        elif isinstance(cond, dict):
            value = metadata.get(key)
            for op, expected in cond.items():
                ok = {
                    "$eq": lambda: value == expected,
                    "$ne": lambda: value != expected,
                    "$in": lambda: value in expected,
                    "$nin": lambda: value not in expected,
                }.get(op, lambda: False)()
                if not ok:
                    return False
        elif metadata.get(key) != cond:
            return False
    return True


class KeywordIndex:
    def __init__(self, chunks: list[Document]):
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks]) if chunks else None

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int, where: dict | None = None) -> list[tuple[Document, float]]:
        """Top-k chunks by BM25 score, restricted to chunks matching `where`."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        results = []
        for i in np.argsort(scores)[::-1]:
            if scores[i] <= 0 or len(results) >= k:
                break
            if matches_filter(self._chunks[i].metadata, where):
                results.append((self._chunks[i], float(scores[i])))
        return results


_index: KeywordIndex | None = None
_lock = threading.Lock()


def get_keyword_index() -> KeywordIndex:
    global _index
    with _lock:
        if _index is None:
            _index = KeywordIndex(get_all_chunks())
        return _index


def invalidate_keyword_index() -> None:
    """Force a rebuild on next use; call after documents are added or removed."""
    global _index
    with _lock:
        _index = None
