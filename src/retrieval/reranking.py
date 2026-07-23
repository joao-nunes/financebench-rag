from __future__ import annotations

from abc import ABC, abstractmethod

from src.evaluation.models import RetrievedDocument
from sentence_transformers import CrossEncoder


class BaseReranker(ABC):
    """Base interface for document rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        ...


class NoOpReranker(BaseReranker):

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        return documents
    

class CrossEncoderReranker(BaseReranker):

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_n: int = 5,
        device: str | None = None,
    ):
        self._model = CrossEncoder(
            model_name,
            device=device,
        )

        self._top_n = top_n

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:

        if not documents:
            return []

        pairs = [
            (
                query,
                doc.page_content,
            )
            for doc in documents
        ]

        scores = self._model.predict(
            pairs,
            convert_to_numpy=True,
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked = []

        for rank, (doc, score) in enumerate(
            ranked[: self._top_n],
            start=1,
        ):

            reranked.append(
                RetrievedDocument(
                    document_id=doc.metadata["document_id"],
                    score=float(score),
                    rank=rank,
                    metadata=doc.metadata,
                )
            )

        return reranked