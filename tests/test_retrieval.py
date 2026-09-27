"""Tests for retrieval: filter building, rank fusion, strategies and filter fallback.

Search backends are replaced with in-memory fakes, so no Chroma or models are needed.
"""

import pytest
from langchain_core.documents import Document

from app.graph.nodes import retrieval
from app.graph.nodes.retrieval import build_where, reciprocal_rank_fusion, retrieve


def doc(chunk_id: str, **meta) -> Document:
    return Document(page_content=f"content of {chunk_id}", metadata={"chunk_id": chunk_id, **meta})


A, B, C, D = doc("a", product="fastapi"), doc("b", product="fastapi"), doc("c", product="redis"), doc("d")


@pytest.fixture
def fake_search(monkeypatch):
    """Semantic ranks a,b,c; keyword ranks c,d,a. Filters apply by exact product match."""
    calls = {"semantic": 0, "keyword": 0}

    def _filter(docs, where):
        if not where:
            return docs
        return [d for d in docs if d.metadata.get("product") == where.get("product")]

    def semantic(query, k, where=None):
        calls["semantic"] += 1
        return _filter([A, B, C], where)[:k]

    def keyword(query, k, where=None):
        calls["keyword"] += 1
        return _filter([C, D, A], where)[:k]

    monkeypatch.setattr(retrieval, "semantic_search", semantic)
    monkeypatch.setattr(retrieval, "keyword_search", keyword)
    return calls


# --- build_where ------------------------------------------------------------

def test_build_where():
    assert build_where(None) is None
    assert build_where({}) is None
    assert build_where({"product": "fastapi"}) == {"product": "fastapi"}
    assert build_where({"product": "fastapi", "version": None}) == {"product": "fastapi"}
    assert build_where({"product": "fastapi", "has_code": True}) == {
        "$and": [{"product": "fastapi"}, {"has_code": True}]
    }
    assert build_where({"version": ["1", "2"]}) == {"version": {"$in": ["1", "2"]}}
    raw = {"$or": [{"product": "a"}, {"product": "b"}]}
    assert build_where(raw) is raw


# --- reciprocal_rank_fusion -------------------------------------------------

def test_rrf_rewards_agreement_between_lists():
    fused = reciprocal_rank_fusion([[A, B, C], [C, D, A]], [0.5, 0.5], rrf_k=10)
    order = [d.metadata["chunk_id"] for d, _, _ in fused]
    # a (ranks 1 and 3) and c (ranks 3 and 1) appear in both lists and beat b and d.
    assert set(order[:2]) == {"a", "c"}
    ranks = {d.metadata["chunk_id"]: r for d, _, r in fused}
    assert ranks["a"] == [1, 3] and ranks["d"] == [None, 2]


def test_rrf_weights_shift_the_winner():
    semantic_heavy = reciprocal_rank_fusion([[A, B], [B, A]], [0.9, 0.1], rrf_k=10)
    keyword_heavy = reciprocal_rank_fusion([[A, B], [B, A]], [0.1, 0.9], rrf_k=10)
    assert semantic_heavy[0][0] is A
    assert keyword_heavy[0][0] is B


# --- retrieve ---------------------------------------------------------------

def test_strategy_none_skips_search(fake_search):
    result = retrieve("hi", strategy="none")
    assert result.chunks == [] and fake_search == {"semantic": 0, "keyword": 0}


def test_semantic_and_keyword_strategies_use_one_backend(fake_search):
    sem = retrieve("q", strategy="semantic", k=3, use_reranker=False)
    assert [c.chunk_id for c in sem.chunks] == ["a", "b", "c"]
    kw = retrieve("q", strategy="keyword", k=3, use_reranker=False)
    assert [c.chunk_id for c in kw.chunks] == ["c", "d", "a"]
    assert fake_search == {"semantic": 1, "keyword": 1}


def test_hybrid_fuses_both_and_records_ranks(fake_search):
    result = retrieve("q", strategy="hybrid", k=4, alpha=0.5, use_reranker=False)
    by_id = {c.chunk_id: c for c in result.chunks}
    assert set(by_id) == {"a", "b", "c", "d"}
    assert (by_id["a"].semantic_rank, by_id["a"].keyword_rank) == (1, 3)
    assert "semantic" in result.timings_ms and "keyword" in result.timings_ms
    assert result.to_dict()["chunks"][0]["chunk_id"] in by_id


def test_filters_are_applied(fake_search):
    result = retrieve("q", filters={"product": "fastapi"}, k=5, use_reranker=False)
    assert {c.chunk_id for c in result.chunks} == {"a", "b"}
    assert result.filter_relaxed is False


def test_filter_fallback_when_nothing_matches(fake_search):
    result = retrieve("q", filters={"product": "kafka"}, k=2, use_reranker=False)
    assert result.filter_relaxed is True
    assert len(result.chunks) == 2


def test_multi_query_searches_each_sub_query(fake_search):
    result = retrieve("q", strategy="multi_query", sub_queries=["q1", "q2"], k=2, use_reranker=False)
    assert result.sub_queries == ["q1", "q2"]
    assert fake_search == {"semantic": 2, "keyword": 2}
    assert len(result.chunks) == 2


def test_reranker_reorders_candidates(fake_search, monkeypatch):
    class FakeReranker:
        def predict(self, pairs):
            return [10.0 if "of d" in text else 0.0 for _, text in pairs]

    monkeypatch.setattr(retrieval, "get_reranker", lambda: FakeReranker())
    result = retrieve("q", k=2, use_reranker=True)
    assert result.reranked is True
    assert result.chunks[0].chunk_id == "d" and result.chunks[0].rerank_score == 10.0
    assert "rerank" in result.timings_ms
