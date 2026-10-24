#!/usr/bin/env python
"""
Build a FAISS vector store from a collection of PDF documents.

Examples
--------
# Full FinanceBench
python scripts/build_vectorstore.py \
    --input data/financebench \
    --output vectorstore/full

# Sample dataset
python scripts/build_vectorstore.py \
    --input data/sample \
    --output vectorstore/sample

# Force rebuilding cached artifacts
python scripts/build_vectorstore.py \
    --input data/sample \
    --output vectorstore/sample \
    --rebuild
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from src.indexing.embedding_engine import EmbeddingEngine
from src.indexing.embeddings import get_embedding_model
from src.indexing.faiss_store import FAISSStore
from src.indexing.indexer import FAISSIndexer
from src.ingestion.helpers import enrich_chunk
from src.ingestion.loaders import load_pdf
from src.ingestion.splitter import split_documents
from src.logging_config import configure_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Build a FAISS vector store.")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Dataset directory (e.g. data/sample or data/financebench).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where the FAISS index will be written.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore cached artifacts.",
    )

    return parser.parse_args()


def cached(path: Path, build, rebuild: bool):
    """Load a pickled result from path, or build and save it."""

    if path.exists() and not rebuild:
        print(f"Loading cached {path.name}...")
        with path.open("rb") as f:
            return pickle.load(f)

    value = build()
    with path.open("wb") as f:
        pickle.dump(value, f)
    return value


def build_chunks(documents):
    start = time.perf_counter()

    chunks = [enrich_chunk(chunk) for chunk in split_documents(documents)]

    print(f"Generated {len(chunks):,} chunks in {time.perf_counter() - start:.1f}s")
    return chunks


def load_documents(pdf_files):
    documents = []
    failed = []

    start = time.perf_counter()

    for pdf in pdf_files:
        print(f"Loading {pdf.name}")

        try:
            documents.extend(load_pdf(pdf))
        except Exception as e:
            print(f"❌ {pdf.name}")
            print(f"   {e}")
            failed.append(pdf.name)

    elapsed = time.perf_counter() - start

    print(f"\nLoaded {len(documents):,} pages in {elapsed:.1f}s")

    if failed:
        print(f"Skipped {len(failed)} PDFs")

    return documents


def main():

    args = parse_args()
    configure_logging(log_file="logs/indexer.log")

    documents_dir = args.input / "pdfs"

    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    pdf_files = sorted(documents_dir.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(f"No PDFs found in {documents_dir}")

    cache_dir = args.output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    documents_cache = cache_dir / "documents.pkl"
    chunks_cache = cache_dir / "chunks.pkl"

    print("=" * 80)
    print(f"Dataset : {args.input}")
    print(f"PDFs    : {len(pdf_files)}")
    print("=" * 80)

    documents = cached(documents_cache, lambda: load_documents(pdf_files), args.rebuild)
    chunks = cached(chunks_cache, lambda: build_chunks(documents), args.rebuild)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    embedding_model = get_embedding_model()

    embedding_engine = EmbeddingEngine(
        embedding_model=embedding_model,
        batch_size=512,
    )

    vector_store = FAISSStore()

    indexer = FAISSIndexer(
        embedding_engine=embedding_engine,
        vector_store=vector_store,
    )

    print("\nBuilding vector store...")
    start = time.perf_counter()

    indexer.build(chunks)

    elapsed = time.perf_counter() - start

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    # Overwrites index.faiss / index.pkl; the cache next to them is kept.
    vector_store.save(args.output)

    print("\nDone!")
    print(f"Chunks       : {len(chunks):,}")
    print(f"Output       : {args.output}")
    print(f"Elapsed time : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
