"""Index documents for RAG pipeline.

Reads chunking geometry from the named strategy in config/config.yaml
instead of relying on hardcoded defaults, so the corpus is actually chunked
the way the configured strategy declares.

Usage (matching the README):
    python scripts/index_documents.py --input data/docs/ --strategy baseline
    python scripts/index_documents.py --input data/docs/ --strategy optimized
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_strategy, list_strategies
from src.ingestion.chunker import chunk_text_for_strategy


def main() -> int:
    parser = argparse.ArgumentParser(description="Index documents for RAG pipeline")
    parser.add_argument("--input", required=True, help="path to a document or directory")
    parser.add_argument("--strategy", default="baseline", help=f"one of {list_strategies()}")
    args = parser.parse_args()

    try:
        resolved = get_strategy(args.strategy)
    except KeyError:
        print(f"unknown strategy {args.strategy!r}; available: {list_strategies()}",
              file=sys.stderr)
        return 2

    source = Path(args.input)
    if not source.exists():
        print(f"input path not found: {source}", file=sys.stderr)
        return 2

    if source.is_file():
        documents = [source]
    else:
        documents = sorted(p for p in source.rglob("*") if p.is_file())

    print(f"Indexing {len(documents)} document(s) with strategy {args.strategy!r}: "
          f"chunk_size={resolved.chunk_size}, overlap={resolved.chunk_overlap}, "
          f"top_k={resolved.top_k}, rerank={resolved.rerank}")

    total_chunks = 0
    for doc in documents:
        try:
            text = doc.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"  skip (unreadable): {doc}: {exc}", file=sys.stderr)
            continue
        chunks = chunk_text_for_strategy(text, args.strategy)
        total_chunks += len(chunks)
        print(f"  {doc}: {len(chunks)} chunks")

    print(f"Document indexing complete: {total_chunks} chunks total")
    return 0


if __name__ == "__main__":
    sys.exit(main())