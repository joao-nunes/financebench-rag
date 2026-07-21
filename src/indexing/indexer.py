from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from src.indexing.embedding_engine import EmbeddingEngine
from src.indexing.faiss_store import FAISSStore
from src.utils.logger import get_logger


logger = get_logger(__name__)


class FAISSIndexer:
    """
    Coordinates the indexing pipeline.

    Responsibilities
    ----------------
    - Iterate over document batches
    - Generate embeddings
    - Insert vectors into FAISS
    - Report progress

    It intentionally knows nothing about the
    implementation details of embeddings or FAISS.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: FAISSStore,
    ) -> None:

        self.embedding_engine = embedding_engine
        self.vector_store = vector_store

    def build(
        self,
        documents: Sequence[Document],
    ) -> FAISSStore:
        """
        Build a FAISS index from a collection of documents.
        """

        if len(documents) == 0:
            raise ValueError("No documents provided.")

        logger.info(
            "Starting indexing of %d documents.",
            len(documents),
        )

        total_docs = len(documents)

        indexed_docs = 0

        start_time = time.perf_counter()

        for batch_number, batch, embeddings in self.embedding_engine.embed(documents):

            if batch_number == 1:

                self.vector_store.create(
                    embeddings=embeddings,
                    documents=batch,
                )

            else:

                self.vector_store.add(
                    embeddings=embeddings,
                    documents=batch,
                )

            indexed_docs += len(batch)

            elapsed = time.perf_counter() - start_time

            throughput = indexed_docs / elapsed

            remaining = total_docs - indexed_docs

            eta = remaining / throughput if throughput > 0 else 0

            logger.info(
                (
                    "[Batch %d] "
                    "Indexed %d/%d documents | "
                    "%.1f docs/s | "
                    "ETA %s | "
                    "Index size: %d"
                ),
                batch_number,
                indexed_docs,
                total_docs,
                throughput,
                timedelta(seconds=int(eta)),
                self.vector_store.size,
            )

        elapsed = time.perf_counter() - start_time

        logger.info("=" * 80)
        logger.info("Indexing completed")
        logger.info("Documents      : %d", total_docs)
        logger.info("Vectors        : %d", self.vector_store.size)
        logger.info(
            "Elapsed        : %s",
            timedelta(seconds=int(elapsed)),
        )
        logger.info(
            "Throughput     : %.1f docs/s",
            total_docs / elapsed,
        )
        logger.info("=" * 80)

        return self.vector_store

    def save(
        self,
        output_dir: str | Path,
    ) -> None:
        """
        Save the current index.
        """
        self.vector_store.save(output_dir)