from __future__ import annotations

import time
from typing import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingEngine:
    """
    Generates embeddings for documents in configurable batches.

    This class is responsible only for embedding generation.
    It has no knowledge of FAISS or vector stores.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        batch_size: int = 512,
    ) -> None:

        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def batches(
        self,
        documents: Sequence[Document],
    ) -> Iterator[list[Document]]:
        """
        Yield batches of documents.
        """

        for start in range(0, len(documents), self.batch_size):
            yield list(documents[start : start + self.batch_size])

    def embed_batch(
        self,
        documents: Sequence[Document],
    ) -> list[list[float]]:
        """
        Generate embeddings for a single batch.
        """

        texts = [doc.page_content for doc in documents]

        return self.embedding_model.embed_documents(texts)

    def embed(
        self,
        documents: Sequence[Document],
    ) -> Iterator[tuple[list[Document], list[list[float]]]]:
        """
        Lazily generate embeddings for all documents.

        Yields
        ------
        tuple[documents, embeddings]
        """

        total_docs = len(documents)

        logger.info(
            "Generating embeddings for %d documents.",
            total_docs,
        )

        start_time = time.perf_counter()

        embedded = 0

        for step, batch in enumerate(self.batches(documents)):

            vectors = self.embed_batch(batch)

            embedded += len(batch)

            elapsed = time.perf_counter() - start_time

            throughput = embedded / elapsed if elapsed > 0 else 0

            logger.info(
                "Embedded %d/%d documents (%.1f docs/s)",
                embedded,
                total_docs,
                throughput,
            )

            yield step, batch, vectors

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Finished embedding %d documents in %.1f seconds "
            "(%.1f docs/s)",
            total_docs,
            elapsed,
            total_docs / elapsed,
        )