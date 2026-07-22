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
        embeddings,
        documents,
    ) -> None:

        if self._store is not None:
            raise RuntimeError("Index already exists.")

        dimension = len(embeddings[0])

        self._store = FAISS(
            embedding_function=None,
            index=faiss.IndexFlatL2(dimension),
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

        text_embeddings = [
            (doc.page_content, emb)
            for doc, emb in zip(documents, embeddings)
        ]


        embeddings = np.asarray(embeddings, dtype=np.float32)

        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 768

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or Inf")

        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN")

        if np.isinf(embeddings).any():
            raise ValueError("Embeddings contain Inf")
        
        logger.info(
            "Batch stats | min=%f max=%f mean=%f",
            embeddings.min(),
            embeddings.max(),
            embeddings.mean(),
        )

        self._store.add_embeddings(
            text_embeddings=text_embeddings,
            metadatas=[doc.metadata for doc in documents],
        )

        logger.info(
            "Added %d vectors (total=%d)",
            len(documents),
            self.size,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._store.save_local(str(path))


    def load(
        self,
        path: str | Path,
        embedding_model,
    ) -> None:

        logger.info("Loading FAISS index...")

        self._store = FAISS.load_local(
            folder_path=str(path),
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )