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
import shutil
import time
from pathlib import Path

from src.indexing.embedding_engine import EmbeddingEngine
from src.indexing.embeddings import get_embedding_model
from src.indexing.faiss_store import FAISSStore
from src.indexing.indexer import FAISSIndexer
from src.ingestion.helpers import enrich_chunk
from src.ingestion.loaders import load_pdf
from src.ingestion.splitter import split_documents
from src.utils.cache import (
    cache_exists,
    load_cache,
    save_cache,
)


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

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    if cache_exists(documents_cache) and not args.rebuild:

        print("Loading cached documents...")
        documents = load_cache(documents_cache)

    else:

        documents = load_documents(pdf_files)
        save_cache(documents, documents_cache)

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    if cache_exists(chunks_cache) and not args.rebuild:

        print("Loading cached chunks...")
        chunks = load_cache(chunks_cache)

    else:

        start = time.perf_counter()

        chunks = split_documents(documents)
        chunks = [enrich_chunk(chunk) for chunk in chunks]

        elapsed = time.perf_counter() - start

        print(f"Generated {len(chunks):,} chunks in {elapsed:.1f}s")

        save_cache(chunks, chunks_cache)

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

    if args.output.exists():
        shutil.rmtree(args.output)

    args.output.mkdir(parents=True, exist_ok=True)

    vector_store.save(args.output)

    print("\nDone!")
    print(f"Chunks       : {len(chunks):,}")
    print(f"Output       : {args.output}")
    print(f"Elapsed time : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
