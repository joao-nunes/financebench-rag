from __future__ import annotations

from abc import ABC, abstractmethod

from src.retrieval.models import RetrievalResult


class Reranker(ABC):
    """Base interface for document rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        ...


