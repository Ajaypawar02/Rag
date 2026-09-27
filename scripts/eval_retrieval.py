"""Benchmark retrieval configurations on the labelled query set.

Each query in data/eval/retrieval_gold.json names the source document and a
snippet the correct chunk must contain. Queries are either "exact" (reuse the
docs' wording/identifiers) or "paraphrase" (natural language, different words).

Metrics: hit@k (a correct chunk is in the top k), MRR (mean reciprocal rank of
the first correct chunk) and p50 latency.

Usage:
    python -m scripts.eval_retrieval            # compare standard configurations
    python -m scripts.eval_retrieval -k 3
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from app.config import PROJECT_ROOT
from app.graph.nodes.retrieval import retrieve

GOLD_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_gold.json"

CONFIGS: dict[str, dict] = {
    "semantic": {"strategy": "semantic", "use_reranker": False},
    "keyword (BM25)": {"strategy": "keyword", "use_reranker": False},
    "hybrid (default)": {"strategy": "hybrid", "use_reranker": False},
    "hybrid + rerank": {"strategy": "hybrid", "use_reranker": True},
}


def first_hit_rank(result, source: str, expected_text: str) -> int | None:
    for rank, chunk in enumerate(result.chunks, start=1):
        if chunk.document.metadata.get("source") == source and expected_text.lower() in chunk.document.page_content.lower():
            return rank
    return None


def evaluate(gold: list[dict], k: int, **config) -> dict:
    ranks, latencies = [], []
    for item in gold:
        result = retrieve(item["query"], k=k, **config)
        ranks.append(first_hit_rank(result, item["source"], item["expected_text"]))
        latencies.append(result.timings_ms["total"])
    return {
        "hit": sum(r is not None for r in ranks) / len(ranks),
        "mrr": statistics.mean(1 / r if r else 0 for r in ranks),
        "p50_ms": statistics.median(latencies),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text())
    subsets = {"all": gold, **{t: [g for g in gold if g["type"] == t] for t in ("exact", "paraphrase")}}
    retrieve("warm up models", use_reranker=True)  # exclude one-off model loading from latency

    print(f"{len(gold)} queries, k={args.k}\n")
    header = f"{'config':<18}" + "".join(f"{name + ' hit/MRR':>22}" for name in subsets) + f"{'p50 ms':>9}"
    print(header)
    print("-" * len(header))
    for name, config in CONFIGS.items():
        scores = {subset: evaluate(items, args.k, **config) for subset, items in subsets.items()}
        cells = "".join(f"{s['hit']:>16.2f} /{s['mrr']:.2f}" for s in scores.values())
        print(f"{name:<18}{cells}{scores['all']['p50_ms']:>9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
