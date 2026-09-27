"""Ingest the sample documentation into the vector store with origin=seed.

Usage:
    python -m scripts.seed_docs            # ingest data/sample_docs/
    python -m scripts.seed_docs --reset    # drop the collection first
    python -m scripts.seed_docs --if-empty # seed only a fresh collection (used by docker-compose)
    python -m scripts.seed_docs --dir path/to/docs
"""

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.ingestion.pipeline import ingest_directory
from app.memory.vectorstore import list_documents, reset_collection


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=settings.sample_docs_dir, help="directory to ingest")
    parser.add_argument("--reset", action="store_true", help="drop the collection before ingesting")
    parser.add_argument("--if-empty", action="store_true", help="do nothing if documents are already indexed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.if_empty and (existing := list_documents()):
        print(f"Collection '{settings.chroma_collection}' already holds {len(existing)} documents; skipping seed.")
        return 0
    if args.reset:
        reset_collection()
        print(f"Reset collection '{settings.chroma_collection}'.")

    results, errors = ingest_directory(args.dir, origin="seed")

    print(f"\n{'file':<36} {'type':<9} {'category':<16} {'product':<11} {'ver':<6} {'chunks':>6} {'code':>5} {'sec':>6}")
    for r in results:
        print(
            f"{r.source:<36} {r.doc_type:<9} {r.doc_category:<16} {r.product or '-':<11} "
            f"{r.version or '-':<6} {r.chunks:>6} {r.chunks_with_code:>5} {r.seconds:>6.2f}"
        )
    for name, err in errors.items():
        print(f"FAILED {name}: {err}")

    total = sum(r.chunks for r in results)
    print(f"\n{len(results)} documents, {total} chunks ingested. Collection now holds {len(list_documents())} documents.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
