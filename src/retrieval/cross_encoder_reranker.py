import logging

from sentence_transformers import CrossEncoder

from src.exceptions import RerankingError
from src.retrieval.models import RetrievalResult
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(Reranker):
    """
    Scores each (query, chunk) pair with a cross-encoder.

    Keeps the top_n chunks, or all of them when top_n is None.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_n: int | None = 5,
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
            scores = self._model.predict(
                [(query, chunk.content) for chunk in chunks],
                convert_to_numpy=True,
            )
            ranked = sorted(
                zip(chunks, scores, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )
            return [
                RetrievalResult(
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=float(score),
                    metadata=chunk.metadata,
                )
                for chunk, score in ranked[: self._top_n]
            ]

        except Exception as e:
            logger.debug(
                "CrossEncoder reranking failed for %d chunks.",
                len(chunks),
                exc_info=True,
            )
            raise RerankingError("Failed to rerank retrieved documents.") from e
