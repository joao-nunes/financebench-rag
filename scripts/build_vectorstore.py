from pathlib import Path

from src.config import (
    CHUNKS_CACHE,
    DOCUMENTS_CACHE,
    PDF_DIR,
    VECTORSTORE_DIR,
    CHECKPOINT_DIR
)

from src.ingestion.loaders import load_pdf
from src.ingestion.splitter import split_documents

from src.indexing.embeddings import get_embedding_model
from src.indexing.embedding_engine import EmbeddingEngine
from src.indexing.faiss_store import FAISSStore
from src.indexing.indexer import FAISSIndexer
from src.indexing.checkpoint import CheckpointManager

from src.utils.cache import (
    cache_exists,
    load_cache,
    save_cache,
)

import argparse
import time


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore cache and rebuild all artifacts."
    )

    args = parser.parse_args()
    pdf_dir = Path(PDF_DIR)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if len(pdf_files) == 0:
        raise RuntimeError(f"No PDFs found in {pdf_dir}")

    print("=" * 80)
    print(f"Found {len(pdf_files)} PDFs")
    print("=" * 80)


    if cache_exists(DOCUMENTS_CACHE) and not args.rebuild:
        print("Loading cached documents...")
        documents = load_cache(DOCUMENTS_CACHE)
    else:
        documents = []
        failed_pdfs = []
        start = time.perf_counter()
        for pdf in pdf_files:
            print(f"Loading {pdf.name}")

            try:
                docs = load_pdf(pdf)
                documents.extend(docs)
            except Exception as e:
                print(f"❌ Failed: {pdf.name}")
                print(f"   {e}")
                failed_pdfs.append((pdf.name, str(e)))

        elapsed = time.perf_counter() - start
        print(f"Loaded {len(documents):,} pages in {elapsed:.1f}s")
        print(f"Skipped {len(failed_pdfs)} PDFs")

        if failed_pdfs:
            print("\nFailed PDFs:")
            for name, err in failed_pdfs:
                print(f" - {name}: {err}")

        save_cache(documents, DOCUMENTS_CACHE)

    if cache_exists(CHUNKS_CACHE) and not args.rebuild:
        print("Loading cached chunks...")
        chunks = load_cache(CHUNKS_CACHE)
    else:
        start = time.perf_counter()
        chunks = split_documents(documents)
        elapsed = time.perf_counter() - start
        print(f"Generated {len(chunks)} chunks in {elapsed:.1f}s")
        save_cache(chunks, CHUNKS_CACHE)

    embedding_model = get_embedding_model()

    embedding_engine = EmbeddingEngine(
        embedding_model=embedding_model,
        batch_size=512,
    )

    vector_store = FAISSStore()

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=CHECKPOINT_DIR,
    )

    indexer = FAISSIndexer(
        embedding_engine=embedding_engine,
        vector_store=vector_store,
    )
    indexer.build(chunks)

    vector_store.save(VECTORSTORE_DIR)


if __name__ == "__main__":
    main()