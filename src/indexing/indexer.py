from __future__ import annotations

import time
from datetime import timedelta
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from tqdm import tqdm


class FAISSIndexer:

    def __init__(
        self,
        embedding_model: Embeddings,
        batch_size: int = 512,
    ):
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def build(
        self,
        documents: Sequence[Document],
    ) -> FAISS:

        if len(documents) == 0:
            raise ValueError("No documents to index.")

        total_docs = len(documents)

        total_batches = (
            total_docs + self.batch_size - 1
        ) // self.batch_size

        start_time = time.perf_counter()

        vectorstore = None

        progress = tqdm(
            range(0, total_docs, self.batch_size),
            total=total_batches,
            desc="Building FAISS",
            unit="batch",
        )

        indexed_docs = 0

        for batch_number, start in enumerate(progress, start=1):

            batch = documents[start:start + self.batch_size]

            if vectorstore is None:
                vectorstore = FAISS.from_documents(
                    batch,
                    self.embedding_model,
                )
            else:
                vectorstore.add_documents(batch)

            indexed_docs += len(batch)

            elapsed = time.perf_counter() - start_time

            throughput = indexed_docs / elapsed

            remaining_docs = total_docs - indexed_docs

            eta_seconds = (
                remaining_docs / throughput
                if throughput > 0
                else 0
            )

            progress.set_postfix(
                docs=f"{indexed_docs:,}/{total_docs:,}",
                speed=f"{throughput:.1f} docs/s",
                eta=str(
                    timedelta(
                        seconds=int(eta_seconds)
                    )
                ),
            )

        elapsed = time.perf_counter() - start_time

        print("\nIndexing completed")
        print("=" * 80)
        print(f"Indexed documents : {indexed_docs:,}")
        print(f"Index size         : {vectorstore.index.ntotal:,}")
        print(f"Elapsed            : {timedelta(seconds=int(elapsed))}")
        print(
            f"Average throughput : "
            f"{indexed_docs / elapsed:.1f} docs/s"
        )
        print("=" * 80)

        return vectorstore