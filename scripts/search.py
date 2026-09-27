"""Try the retriever from the command line (requires Chroma to be running and seeded).

Usage:
    python -m scripts.search "How do I declare a dependency?"
    python -m scripts.search "model_validate" --strategy keyword --filter product=pydantic
    python -m scripts.search "rollback" --strategy semantic --rerank -k 3
    python -m scripts.search "pydantic validators" --strategy multi_query \
        --sub-query "field_validator decorator" --sub-query "model_validator"
    python -m scripts.search "..." --json     # full diagnostics as JSON
"""

import argparse
import json
import logging
import sys

from app.graph.nodes.retrieval import retrieve


def _parse_filter(item: str) -> tuple[str, object]:
    key, _, value = item.partition("=")
    if value.lower() in ("true", "false"):
        return key, value.lower() == "true"
    return key, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--strategy", default="hybrid", choices=["hybrid", "semantic", "keyword", "multi_query"])
    parser.add_argument("--filter", action="append", default=[], metavar="FIELD=VALUE",
                        help="metadata filter, repeatable (e.g. product=fastapi, has_code=true)")
    parser.add_argument("--sub-query", action="append", default=[], help="sub-query for multi_query, repeatable")
    parser.add_argument("-k", type=int, default=None, help="number of results")
    parser.add_argument("--alpha", type=float, default=None, help="semantic weight 0..1 for hybrid")
    rerank = parser.add_mutually_exclusive_group()
    rerank.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                        help="force cross-encoder reranking on (default: RERANK_ENABLED)")
    rerank.add_argument("--no-rerank", dest="rerank", action="store_false", help="force reranking off")
    parser.add_argument("--json", action="store_true", help="print full result as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    result = retrieve(
        args.query,
        strategy=args.strategy,
        filters=dict(_parse_filter(f) for f in args.filter) or None,
        sub_queries=args.sub_query or None,
        k=args.k,
        alpha=args.alpha,
        use_reranker=args.rerank,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"strategy={result.strategy} filters={result.filters or '-'} relaxed={result.filter_relaxed} "
          f"reranked={result.reranked} timings_ms={result.timings_ms}\n")
    for i, c in enumerate(result.chunks, start=1):
        m = c.document.metadata
        where = m.get("section") or (f"page {m['page']}" if "page" in m else "-")
        print(f"{i}. score={c.score:.3f}  sem#{c.semantic_rank or '-'}  kw#{c.keyword_rank or '-'}  "
              f"{m.get('source')} :: {where}")
        preview = c.document.page_content.split("\n\n", 1)[-1].replace("\n", " ")
        print(f"   {preview[:160]}...\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
