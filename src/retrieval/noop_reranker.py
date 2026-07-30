from src.retrieval.reranker import Reranker
from src.retrieval.models import RetrievalResult


class NoOpReranker(Reranker):

    def rerank(
        self,
        query: str,
        chunks: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        return chunks