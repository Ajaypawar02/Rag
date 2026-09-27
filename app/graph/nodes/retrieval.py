"""Retrieval executors: semantic, keyword, hybrid and multi-query search with reranking.

Pipeline for one `retrieve()` call:

    query ──┬─> semantic search (Chroma, embeddings) ──┐
            └─> keyword search  (BM25)  ───────────────┴─> weighted RRF fusion
                                                        ─> cross-encoder rerank ─> top-k

- Hybrid fusion uses Reciprocal Rank Fusion: each method contributes
  weight / (rrf_k + rank). Ranks, not raw scores, are fused because cosine
  similarity and BM25 scores live on incomparable scales.
- `multi_query` runs hybrid search per sub-query (from query decomposition or
  rewriting) and fuses those lists the same way.
- Metadata filters (product, version, doc_type, has_code, ...) come from the
  Retrieval Router. If they match nothing, the search is retried without them
  and the result is flagged `filter_relaxed` so the agent can say so.

Every result records per-method ranks and scores plus stage latencies, which
feed the agent decision log and the explainability dashboard.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langsmith import traceable

from app.config import get_settings
from app.llm import get_reranker
from app.memory.keyword_index import get_keyword_index
from app.memory.vectorstore import get_vectorstore
from app.observability.decision_log import decision, traced

Strategy = Literal["hybrid", "semantic", "keyword", "multi_query", "none"]


@dataclass
class RetrievedChunk:
    document: Document
    score: float  # final ordering score (rerank score if reranked, else fused score)
    fused_score: float = 0.0
    semantic_rank: int | None = None  # 1-based rank in the semantic list, None if absent
    keyword_rank: int | None = None
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return chunk_key(self.document)

    def to_dict(self) -> dict:
        m = self.document.metadata
        return {
            "chunk_id": self.chunk_id,
            "source": m.get("source"),
            "title": m.get("title"),
            "section": m.get("section"),
            "page": m.get("page"),
            "score": round(self.score, 4),
            "fused_score": round(self.fused_score, 5),
            "semantic_rank": self.semantic_rank,
            "keyword_rank": self.keyword_rank,
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
        }


@dataclass
class RetrievalResult:
    query: str
    strategy: Strategy
    chunks: list[RetrievedChunk] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    filter_relaxed: bool = False
    reranked: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def documents(self) -> list[Document]:
        return [c.document for c in self.chunks]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "sub_queries": self.sub_queries,
            "filters": self.filters,
            "filter_relaxed": self.filter_relaxed,
            "reranked": self.reranked,
            "timings_ms": self.timings_ms,
            "chunks": [c.to_dict() for c in self.chunks],
        }


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

def chunk_key(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.id or doc.page_content[:200]


def build_where(filters: dict[str, Any] | None) -> dict | None:
    """Convert simple `{"field": value}` filters into a Chroma `where` clause.

    Chroma needs an explicit `$and` for more than one condition; list values
    become `$in`. Clauses that already use operators pass through unchanged.
    """
    if not filters:
        return None
    if any(k.startswith("$") for k in filters):
        return filters
    clauses = [{k: {"$in": v} if isinstance(v, list) else v} for k, v in filters.items() if v is not None]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def semantic_search(query: str, k: int, where: dict | None = None) -> list[Document]:
    """Top-k chunks by embedding similarity."""
    return get_vectorstore().similarity_search(query, k=k, filter=where)


def keyword_search(query: str, k: int, where: dict | None = None) -> list[Document]:
    """Top-k chunks by BM25 score."""
    return [doc for doc, _ in get_keyword_index().search(query, k=k, where=where)]


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], weights: list[float], rrf_k: int
) -> list[tuple[Document, float, list[int | None]]]:
    """Fuse ranked lists: score(d) = sum_i weight_i / (rrf_k + rank_i(d)).

    Returns (document, fused_score, [rank in each list or None]) sorted by score.
    """
    docs: dict[str, Document] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, list[int | None]] = {}
    for list_idx, (ranked, weight) in enumerate(zip(ranked_lists, weights)):
        for rank, doc in enumerate(ranked, start=1):
            key = chunk_key(doc)
            docs.setdefault(key, doc)
            ranks.setdefault(key, [None] * len(ranked_lists))[list_idx] = rank
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [(docs[key], scores[key], ranks[key]) for key in ordered]


def hybrid_search(
    query: str, k: int, where: dict | None, alpha: float, timings: dict[str, float] | None = None
) -> list[RetrievedChunk]:
    """Semantic + BM25 search fused with weighted RRF (`alpha` = semantic weight)."""
    s = get_settings()
    timings = timings if timings is not None else {}

    semantic: list[Document] = []
    keyword: list[Document] = []
    if alpha > 0:
        t = time.perf_counter()
        semantic = semantic_search(query, k, where)
        timings["semantic"] = timings.get("semantic", 0.0) + _ms_since(t)
    if alpha < 1:
        t = time.perf_counter()
        keyword = keyword_search(query, k, where)
        timings["keyword"] = timings.get("keyword", 0.0) + _ms_since(t)

    fused = reciprocal_rank_fusion([semantic, keyword], [alpha, 1 - alpha], s.rrf_k)
    return [
        RetrievedChunk(document=doc, score=score, fused_score=score, semantic_rank=r[0], keyword_rank=r[1])
        for doc, score, r in fused
    ]


def rerank(query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """Rescore candidates with a cross-encoder that reads query and chunk together."""
    if not chunks:
        return []
    scores = get_reranker().predict([(query, c.document.page_content) for c in chunks])
    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)
        chunk.score = float(score)
    return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_n]


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _trace_outputs(result: "RetrievalResult") -> dict:
    """Shape the result like a LangChain retriever run so LangSmith renders the documents."""
    return {
        "documents": [
            {"type": "Document", "page_content": c.document.page_content,
             "metadata": {**c.document.metadata, "score": c.score, "semantic_rank": c.semantic_rank,
                          "keyword_rank": c.keyword_rank}}
            for c in result.chunks
        ],
        "strategy": result.strategy,
        "filter_relaxed": result.filter_relaxed,
        "timings_ms": result.timings_ms,
    }


@traceable(run_type="retriever", name="hybrid_retrieval", process_outputs=_trace_outputs)
def retrieve(
    query: str,
    strategy: Strategy = "hybrid",
    filters: dict[str, Any] | None = None,
    sub_queries: list[str] | None = None,
    k: int | None = None,
    alpha: float | None = None,
    use_reranker: bool | None = None,
) -> RetrievalResult:
    """Run one retrieval with the given strategy and return ranked chunks plus diagnostics.

    Args:
        query: The standalone (rewritten) user query; reranking always scores against it.
        strategy: "hybrid" (default), "semantic", "keyword", "multi_query" or "none".
        filters: Metadata constraints, e.g. {"product": "pydantic", "version": "2"}.
        sub_queries: Queries to search for `multi_query`; defaults to [query].
        k: Number of chunks to return (default RETRIEVAL_TOP_K).
        alpha: Semantic weight for fusion (default HYBRID_ALPHA).
        use_reranker: Override RERANK_ENABLED.
    """
    s = get_settings()
    k = k or s.retrieval_top_k
    alpha = s.hybrid_alpha if alpha is None else alpha
    use_reranker = s.rerank_enabled if use_reranker is None else use_reranker
    result = RetrievalResult(query=query, strategy=strategy, filters=filters or {})
    if strategy == "none":
        return result

    start = time.perf_counter()
    method_alpha = {"semantic": 1.0, "keyword": 0.0}.get(strategy, alpha)
    queries = (sub_queries or [query]) if strategy == "multi_query" else [query]
    result.sub_queries = queries if strategy == "multi_query" else []

    def search(where: dict | None) -> list[RetrievedChunk]:
        per_query = [hybrid_search(q, s.retrieval_fetch_k, where, method_alpha, result.timings_ms) for q in queries]
        if len(per_query) == 1:
            return per_query[0]
        fused = reciprocal_rank_fusion(
            [[c.document for c in lst] for lst in per_query], [1.0] * len(per_query), s.rrf_k
        )
        best = {c.chunk_id: c for lst in per_query for c in lst}  # keep per-method ranks for diagnostics
        return [
            RetrievedChunk(
                document=doc, score=score, fused_score=score,
                semantic_rank=best[chunk_key(doc)].semantic_rank, keyword_rank=best[chunk_key(doc)].keyword_rank,
            )
            for doc, score, _ in fused
        ]

    candidates = search(build_where(filters))
    if not candidates and filters and s.filter_fallback:
        candidates = search(None)
        result.filter_relaxed = True

    if use_reranker and candidates:
        t = time.perf_counter()
        result.chunks = rerank(query, candidates[: s.retrieval_fetch_k], k)
        result.timings_ms["rerank"] = _ms_since(t)
        result.reranked = True
    else:
        result.chunks = candidates[:k]

    result.timings_ms["total"] = _ms_since(start)
    return result


# --------------------------------------------------------------------------- #
# Graph node
# --------------------------------------------------------------------------- #

def to_context(chunk: RetrievedChunk) -> dict:
    """JSON-friendly view of a chunk for graph state (see state.RetrievedContext)."""
    m = chunk.document.metadata
    context = {
        "chunk_id": chunk.chunk_id,
        "content": chunk.document.page_content,
        "source": m.get("source"),
        "title": m.get("title"),
        "section": m.get("section"),
        "page": m.get("page"),
        "origin": m.get("origin"),
        "score": round(chunk.score, 4),
    }
    return {k: v for k, v in context.items() if v is not None}


@traced("retrieval")
def retrieval_node(state: dict) -> dict:
    """Execute the router's plan against the knowledge base."""
    query = state.get("rewritten_query") or state.get("query", "")
    result = retrieve(
        query,
        strategy=state.get("retrieval_strategy", "hybrid"),
        filters=state.get("filters") or None,
        sub_queries=state.get("sub_queries") or None,
    )
    sources = sorted({c.document.metadata.get("source", "?") for c in result.chunks})
    reasoning = "Filters matched nothing; searched all documents instead." if result.filter_relaxed else ""
    return {
        "retrieved": [to_context(c) for c in result.chunks],
        "retrieval_info": result.to_dict(),
        "decisions": [
            decision(f"{len(result.chunks)} chunks from {', '.join(sources) or 'no documents'}", reasoning,
                     filter_relaxed=result.filter_relaxed, timings_ms=result.timings_ms,
                     chunks=[c.to_dict() for c in result.chunks])
        ],
    }
