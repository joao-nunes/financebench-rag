from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from src.indexing.embedding_engine import EmbeddingEngine
from src.indexing.faiss_store import FAISSStore
from src.indexing.checkpoint import CheckpointManager
from src.indexing.metadata import CheckpointMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSIndexer:
    """
    Orchestrates the complete indexing pipeline.

        Documents
            │
            ▼
    EmbeddingEngine
            │
            ▼
    FAISSStore
            │
            ▼
    CheckpointManager
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: FAISSStore,
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_every: int = 20,
    ) -> None:

        self.embedding_engine = embedding_engine
        self.vector_store = vector_store

        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_every = checkpoint_every

    def build(
        self,
        documents: Sequence[Document],
        resume: bool = False,
    ) -> FAISSStore:
        """
        Build (or resume) a FAISS index.
        """

        if len(documents) == 0:
            raise ValueError("No documents provided.")

        total_documents = len(documents)

        indexed_documents = 0

        first_batch = True

        #
        # Resume from checkpoint
        #
        if (
            resume
            and self.checkpoint_manager is not None
            and self.checkpoint_manager.exists()
        ):

            logger.info("Loading checkpoint...")

            metadata = self.checkpoint_manager.load_metadata()

            self.vector_store = self.checkpoint_manager.load_store(
                self.vector_store,
                self.embedding_engine.embedding_model,
            )

            indexed_documents = metadata.indexed_documents

            documents = documents[indexed_documents:]

            first_batch = False

            logger.info(
                "Resuming from document %d.",
                indexed_documents,
            )

        logger.info(
            "Starting indexing (%d documents).",
            total_documents,
        )

        start_time = time.perf_counter()

        #
        # Main indexing loop
        #
        for batch_idx, (batch, embeddings) in enumerate(
            self.embedding_engine.embed(documents),
            start=1,
        ):

            if first_batch:

                self.vector_store.create(
                    embeddings=embeddings,
                    documents=batch,
                )

                first_batch = False

            else:

                self.vector_store.add(
                    embeddings=embeddings,
                    documents=batch,
                )

            indexed_documents += len(batch)

            elapsed = time.perf_counter() - start_time

            throughput = (
                indexed_documents / elapsed
                if elapsed > 0
                else 0
            )

            remaining = total_documents - indexed_documents

            eta = (
                remaining / throughput
                if throughput > 0
                else 0
            )

            logger.info(
                (
                    "Batch %d | "
                    "%d/%d docs | "
                    "%.1f docs/s | "
                    "ETA %s | "
                    "Index size=%d"
                ),
                batch_idx,
                indexed_documents,
                total_documents,
                throughput,
                timedelta(seconds=int(eta)),
                self.vector_store.size,
            )

            #
            # Save checkpoint
            #
            if (
                self.checkpoint_manager is not None
                and batch_idx % self.checkpoint_every == 0
            ):

                metadata = CheckpointMetadata.create(
                    current_batch=batch_idx,
                    indexed_documents=indexed_documents,
                    total_documents=total_documents,
                    batch_size=self.embedding_engine.batch_size,
                    embedding_model=type(
                        self.embedding_engine.embedding_model
                    ).__name__,
                )

                self.checkpoint_manager.save(
                    self.vector_store,
                    metadata,
                )

        elapsed = time.perf_counter() - start_time

        logger.info("=" * 80)
        logger.info("Indexing complete")
        logger.info("Documents : %d", indexed_documents)
        logger.info("Vectors   : %d", self.vector_store.size)
        logger.info(
            "Elapsed   : %s",
            timedelta(seconds=int(elapsed)),
        )
        logger.info(
            "Speed     : %.1f docs/s",
            indexed_documents / elapsed,
        )
        logger.info("=" * 80)

        #
        # Final checkpoint
        #
        if self.checkpoint_manager is not None:

            metadata = CheckpointMetadata.create(
                current_batch=-1,
                indexed_documents=indexed_documents,
                total_documents=total_documents,
                batch_size=self.embedding_engine.batch_size,
                embedding_model=type(
                    self.embedding_engine.embedding_model
                ).__name__,
            )

            self.checkpoint_manager.save(
                self.vector_store,
                metadata,
            )

        return self.vector_store

    def save(
        self,
        output_dir: str | Path,
    ) -> None:
        """
        Save the final FAISS index.
        """
        self.vector_store.save(output_dir)