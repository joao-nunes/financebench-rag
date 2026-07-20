from pathlib import Path

from src.config import (
    CHUNKS_CACHE,
    DOCUMENTS_CACHE,
    PDF_DIR,
    VECTORSTORE_DIR,
)

from src.ingestion.loaders import load_pdf
from src.ingestion.splitter import split_documents

from src.indexing.embeddings import get_embedding_model
from src.indexing.vectorstore import (
    create_vectorstore,
    save_vectorstore,
)

from src.utils.cache import (
    cache_exists,
    load_cache,
    save_cache,
)

import argparse


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

        for pdf in pdf_files:
            print(f"Loading {pdf.name}")

            try:
                docs = load_pdf(pdf)
                documents.extend(docs)
            except Exception as e:
                print(f"❌ Failed: {pdf.name}")
                print(f"   {e}")
                failed_pdfs.append((pdf.name, str(e)))

        print(f"\nLoaded {len(documents)} pages")
        print(f"Skipped {len(failed_pdfs)} PDFs")

        if failed_pdfs:
            print("\nFailed PDFs:")
            for name, err in failed_pdfs:
                print(f" - {name}: {err}")

    print()
    print(f"Loaded {len(documents)} pages")

    if cache_exists(CHUNKS_CACHE):
        print("Loading cached chunks...")
        chunks = load_cache(CHUNKS_CACHE)
    else:
        chunks = split_documents(documents)
        save_cache(chunks, CHUNKS_CACHE)

    print(f"Generated {len(chunks)} chunks")

    embedding_model = get_embedding_model()

    print("Creating FAISS index...")

    vectorstore = create_vectorstore(
        documents=chunks,
        embedding_model=embedding_model,
    )

    print(f"Saving vector store to {VECTORSTORE_DIR}")

    save_vectorstore(
        vectorstore=vectorstore,
        vectorstore_path=Path(VECTORSTORE_DIR),
    )

    print()
    print("=" * 80)
    print("Finished!")
    print(f"Indexed {vectorstore.index.ntotal} vectors")
    print("=" * 80)


if __name__ == "__main__":
    main()