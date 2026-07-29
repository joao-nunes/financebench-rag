from __future__ import annotations

from abc import ABC, abstractmethod

from src.retrieval.models import RetrievalResult
from sentence_transformers import CrossEncoder
from src.logging_config import logging
from src.exceptions import RerankingError


logger = logging.getLogger(__name__)

class BaseReranker(ABC):
    """Base interface for document rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        ...


class NoOpReranker(BaseReranker):

    def rerank(
        self,
        query: str,
        chunks: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        return chunks
    

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
        chunks: list[RetrievalResult],
    ) -> list[RetrievalResult]:

        if not chunks:
            return []
        try:
            pairs = [
                (
                    query,
                    chunk.content,
                )
                for chunk in chunks
            ]

            scores = self._model.predict(
                pairs,
                convert_to_numpy=True,
            )

            ranked = sorted(
                zip(chunks, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            reranked = []

            for _, (chunk, score) in enumerate(
                ranked[: self._top_n],
                start=1,
            ):
                reranked.append(
                    RetrievalResult(
                        document_id=chunk.document_id,
                        content=chunk.content,
                        score=float(score),
                        metadata=chunk.metadata,
                    )
                )

            return reranked
        
        except Exception as e:
            logger.debug(
                "CrossEncoder reranking failed for %d chunks.",
                len(chunks),
                exc_info=True,
            )
            raise RerankingError("Failed to rerank retrieved documents.") from e