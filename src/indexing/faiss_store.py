from __future__ import annotations

import uuid
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSStore:

    def __init__(self):

        self._store: FAISS | None = None

    @property
    def store(self) -> FAISS:
        if self._store is None:
            raise RuntimeError("Index has not been initialized.")
        return self._store

    @property
    def size(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def create(
        self,
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[Document],
    ) -> None:

        if self._store is not None:
            raise RuntimeError("Index already exists.")

        dimension = len(embeddings[0])

        index = faiss.IndexFlatL2(dimension)

        self._store = FAISS(
            embedding_function=None,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        self.add(embeddings, documents)

    def add(
        self,
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[Document],
    ) -> None:

        if self._store is None:
            raise RuntimeError("Index has not been initialized.")

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        start_idx = self.size

        #
        # Add vectors
        #
        self._store.index.add(embeddings)

        #
        # Add documents
        #
        for i, document in enumerate(documents):

            doc_id = str(uuid.uuid4())

            self._store.docstore.add(
                {
                    doc_id: document
                }
            )

            self._store.index_to_docstore_id[
                start_idx + i
            ] = doc_id

        logger.info(
            "Added %d vectors (total=%d)",
            len(documents),
            self.size,
        )
    def save(
        self,
        path: str | Path,
    ) -> None:

        logger.info("Saving FAISS index...")

        self.store.save_local(str(path))
    
    def load(
        self,
        path: str | Path,
        embedding_model,
    ) -> None:

        logger.info("Loading FAISS index...")

        self._store = FAISS.load_local(
            folder_path=str(path),
            embeddings=embedding_model,
            allow_dangerous_deserialization=False,
        )